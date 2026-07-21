import base64
import os
from io import BytesIO
from types import SimpleNamespace
from openai import OpenAI
from PIL import Image as PILImage
import torch
try:
    from diffusers import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS
    from diffusers.utils import load_image
except RuntimeError as e:
    print(f"Error importing diffusers, due to error: \n\n{e}\n")
    FluxPipeline = None
    PREFERRED_KONTEXT_RESOLUTIONS = dict()

from digital_cousins.utils.python_utils import assert_valid_key
from digital_cousins.models.remote_cache import RemoteModelCache, image_digests


genai = SimpleNamespace(Client=None)
_GOOGLE_GENAI_IMPORTED = False


def _has_patched_genai_client():
    return not _GOOGLE_GENAI_IMPORTED and getattr(genai, "Client", None) is not None


def _get_genai_module():
    global genai, _GOOGLE_GENAI_IMPORTED
    if _has_patched_genai_client():
        return genai
    if not _GOOGLE_GENAI_IMPORTED:
        try:
            from google import genai as google_genai
        except Exception as exc:
            raise RuntimeError(
                "Failed to import google.genai. Install a compatible Google GenAI SDK "
                "or run with TEST_MODE using a cached response."
            ) from exc
        genai = google_genai
        _GOOGLE_GENAI_IMPORTED = True
    return genai


class _FallbackGeminiPart:
    @classmethod
    def from_text(cls, text):
        return SimpleNamespace(text=text, inline_data=None)

    @classmethod
    def from_bytes(cls, data, mime_type):
        return SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=_coerce_bytes(data), mime_type=mime_type),
        )


class _FallbackGeminiContent(SimpleNamespace):
    pass


class _FallbackGeminiConfig(SimpleNamespace):
    pass


class _FallbackGeminiSafetySetting(SimpleNamespace):
    pass


class _FallbackGeminiHttpOptions(SimpleNamespace):
    pass


def _get_genai_types(allow_fallback=False):
    try:
        from google.genai.types import Content, GenerateContentConfig, HttpOptions, Part, SafetySetting
    except Exception as exc:
        if allow_fallback:
            return SimpleNamespace(
                Content=_FallbackGeminiContent,
                GenerateContentConfig=_FallbackGeminiConfig,
                HttpOptions=_FallbackGeminiHttpOptions,
                Part=_FallbackGeminiPart,
                SafetySetting=_FallbackGeminiSafetySetting,
            )
        raise RuntimeError(
            "Failed to import google.genai.types. Install a compatible Google GenAI SDK "
            "or run with TEST_MODE using a cached response."
        ) from exc
    return SimpleNamespace(
        Content=Content,
        GenerateContentConfig=GenerateContentConfig,
        HttpOptions=HttpOptions,
        Part=Part,
        SafetySetting=SafetySetting,
    )


def load_api_keys(path=None):
    """Load KEY=VALUE pairs from api_keys.txt into os.environ (no-op if already set)."""
    from pathlib import Path
    if path is None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "api_keys.txt"
            if candidate.exists():
                path = candidate
                break
    if path is None:
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


load_api_keys()


def _remote_timeout_ms(*env_names, default_ms):
    for env_name in env_names:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            timeout_ms = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer number of milliseconds, got {raw_value!r}") from exc
        return timeout_ms if timeout_ms > 0 else None
    return default_ms


def _get_vertex_imagen_api():
    try:
        import vertexai
        from vertexai.preview.vision_models import Image as VertexImage
        from vertexai.preview.vision_models import ImageGenerationModel, MaskReferenceImage, RawReferenceImage
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Vertex AI image generation dependencies. Install google-cloud-aiplatform "
            "or run with TEST_MODE using a cached response."
        ) from exc
    return SimpleNamespace(
        ImageGenerationModel=ImageGenerationModel,
        MaskReferenceImage=MaskReferenceImage,
        RawReferenceImage=RawReferenceImage,
        VertexImage=VertexImage,
        vertexai=vertexai,
    )


class VLM_API:
    """
    Class for interfacing with remote VLM APIs, e.g.: ChatGPT, Gemini, etc.
    """
    VERSIONS = None

    @staticmethod
    def encode_image(image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            bytes: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class _CachedInlineData:
    def __init__(self, data, mime_type="image/png"):
        self.data = data
        self.mime_type = mime_type

    def __bool__(self):
        return bool(self.data)


class _CachedGeminiPart:
    def __init__(self, payload):
        self.text = payload.get("text")
        inline_data = payload.get("inline_data")
        self.inline_data = None if inline_data is None else _CachedInlineData(
            data=base64.b64decode(inline_data["data_base64"]),
            mime_type=inline_data.get("mime_type", "image/png"),
        )


class _CachedGeminiContent:
    def __init__(self, parts):
        self.parts = [_CachedGeminiPart(part) for part in parts]


class _CachedGeminiCandidate:
    def __init__(self, parts):
        self.content = _CachedGeminiContent(parts)


class _CachedGeminiChunk:
    def __init__(self, payload):
        self.text = payload.get("text", "")
        self.candidates = [_CachedGeminiCandidate(payload.get("parts", []))]


class _CachedImageDatum:
    def __init__(self, b64_json):
        self.b64_json = b64_json


class _CachedImageResponse:
    def __init__(self, images_base64):
        self.data = [_CachedImageDatum(b64_json=image_base64) for image_base64 in images_base64]


class _CachedImagenResult:
    def __init__(self, image_base64):
        image = PILImage.open(BytesIO(base64.b64decode(image_base64)))
        image.load()
        self._pil_image = image


def _coerce_bytes(data):
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception:
            return data.encode("utf-8")
    return bytes(data)


def _pil_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _serialize_gemini_result(result):
    chunks = []
    for res in result or []:
        chunk = {
            "text": getattr(res, "text", "") or "",
            "parts": [],
        }
        candidates = getattr(res, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            for part in getattr(content, "parts", []) or []:
                part_payload = {}
                part_text = getattr(part, "text", None)
                if part_text is not None:
                    part_payload["text"] = part_text
                inline_data = getattr(part, "inline_data", None)
                if inline_data:
                    part_payload["inline_data"] = {
                        "data_base64": base64.b64encode(_coerce_bytes(getattr(inline_data, "data", None))).decode("ascii"),
                        "mime_type": getattr(inline_data, "mime_type", "image/png"),
                    }
                if part_payload:
                    chunk["parts"].append(part_payload)
        chunks.append(chunk)
    return {"chunks": chunks}


def _deserialize_gemini_result(payload):
    return [_CachedGeminiChunk(chunk) for chunk in payload["chunks"]]


def _serialize_gpt_image_result(result):
    return {
        "images_base64": [dat.b64_json for dat in result.data],
    }


def _deserialize_gpt_image_result(payload):
    return _CachedImageResponse(images_base64=payload["images_base64"])


def _serialize_imagen_result(result):
    return {
        "images_base64": [_pil_image_to_base64(res._pil_image) for res in result],
    }


def _deserialize_imagen_result(payload):
    return [_CachedImagenResult(image_base64=image_base64) for image_base64 in payload["images_base64"]]


class Gemini(VLM_API):
    """
    Class for interfacing with supported Gemini models
    """
    # TODO: Is this actually accurate?
    RESOLUTIONS = {
        "1:1": (1024, 1024),
        "3:4": (864, 1184),
        "4:3": (1184, 864),
        "9:16": (736, 1408),
        "16:9": (1408, 736),
    }
    IMAGE_SHAPES = {res for res in RESOLUTIONS.values()}

    # TODO: we need a central location for all models
    VERSIONS = {
        "gemini-2.0-flash-preview-image-generation": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 8192,
        },
        "gemini-2.5-pro": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash-image-preview": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-2.5-flash-image": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-3-pro": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-pro-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3.1-pro-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-flash-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-pro-image-preview": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-3.0-flash": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
    }
    def __init__(
        self,
        project,
        location="global",
        model="gemini-3-pro-image-preview",
        verbose=False,
        timeout_ms=None,
    ):
        """
        Args:
            project (str): Name of the project to use when calling the Gemini client
            location (str): Location to use when calling the Gemini client
            model (str): Gemini model to use. Must be one of self.VERSIONS
            timeout_ms (None or int): Request timeout in milliseconds. If None, uses CDC_GEMINI_TIMEOUT_MS,
                CDC_REMOTE_TIMEOUT_MS, or the default.
        """
        self.project = project
        self.verbose = verbose
        if self.verbose:
            print("="*100)
            print(f"USING PROJECT: {self.project}")
            print("="*100)
        self.location = location
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Gemini model")
        self.model = model
        self.client = None
        self.timeout_ms = (
            timeout_ms
            if timeout_ms is not None
            else _remote_timeout_ms("CDC_GEMINI_TIMEOUT_MS", "CDC_REMOTE_TIMEOUT_MS", default_ms=300_000)
        )


    def __call__(
        self,
        prompt,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_paths (None or str or list of str): If specified, absolute path(s) corresponding to reference image(s)
                to use as part of the overall prompt
            temperature (float): Temperature of the model when querying. Lower values correspond to more deterministic
                outputs
            top_p (float): Determines the cumulative probability of top-p tokens to select from probabilistically.
                E.g.: If top_p=0.7 and tokens a, b, c have probabilities of 0.4, 0.3, 0.2 respectively, only tokens
                a and b will be sampled from
            seed (int): Random seed to use
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            None or list of google.genai.types.GenerateContentResponse: Stream of responses generated from Gemini
        """
        image_paths = [] if image_paths is None else [image_paths] if isinstance(image_paths, (str, os.PathLike)) else list(image_paths)
        cache = RemoteModelCache.from_env()
        cache_request = {
            "prompt": prompt,
            "image_inputs": image_digests(image_paths),
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_output_tokens": self.VERSIONS[self.model]["max_tokens"],
            "response_modalities": self.VERSIONS[self.model]["modalities"],
        }
        cache_key = cache.key_for(provider="gemini", model=self.model, request=cache_request)
        if cache.test_enabled:
            return _deserialize_gemini_result(cache.load_response(provider="gemini", key=cache_key))
        if cache.cache_enabled:
            cached_response = cache.load_response_if_exists(provider="gemini", key=cache_key)
            if cached_response is not None:
                return _deserialize_gemini_result(cached_response)

        genai_types = _get_genai_types(allow_fallback=_has_patched_genai_client())
        parts = [genai_types.Part.from_text(text=prompt)]
        if image_paths:
            msg1_images = []
            for image_path in image_paths:
                msg1_images.append(genai_types.Part.from_bytes(
                    data=self.encode_image(image_path),
                    mime_type="image/png",
                ))
            parts = msg1_images + parts
        contents = [genai_types.Content(
            role="user",
            parts=parts,
        )]

        generate_content_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_output_tokens=self.VERSIONS[self.model]["max_tokens"],
            response_modalities=self.VERSIONS[self.model]["modalities"],
            safety_settings=[genai_types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_HATE",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_HARASSMENT",
                threshold="OFF"
            ), genai_types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
                threshold="OFF"
            )],
        )

        result = None
        if self.client is None:
            client_kwargs = {
                "vertexai": True,
                "project": self.project,
                "location": self.location,
            }
            if self.timeout_ms is not None:
                client_kwargs["http_options"] = genai_types.HttpOptions(timeout=self.timeout_ms)
            self.client = _get_genai_module().Client(**client_kwargs)
        for i in range(n_retries):
            if result is not None:
                break
            if self.verbose:
                print(f"Querying Gemini [{self.model}]: attempt {i + 1} of {n_retries}...", flush=True)
            _result = []
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if print_results:
                        print(chunk.text, end="")
                    _result.append(chunk)
                result = _result
            except Exception as e:
                print(f"\nFailed attempt {i + 1} of {n_retries}: {e}", flush=True)
                print(f"\nFailed attempt {i + 1} of {n_retries}", flush=True)

        print()
        if cache.cache_enabled and result is not None:
            cache.store_response(
                provider="gemini",
                model=self.model,
                key=cache_key,
                request=cache_request,
                response=_serialize_gemini_result(result),
            )
        return result

    def get_result_text(self, result):
        return "".join(res.text for res in result)

    def get_result_images(self, result):
        images = []
        for res in result:
            for part in res.candidates[0].content.parts:
                if part.inline_data:
                    # The image data is in base64 encoded format within part.inline_data.data
                    image_data = part.inline_data.data

                    # You can then process this data, for example, save it as an image file
                    # Decode the base64 data and open it with PIL (Pillow)
                    image = PILImage.open(BytesIO(image_data))
                    images.append(image)
        return images


class Imagen3(VLM_API):
    """
    Class for interfacing with supported Imagen3 models
    """
    VERSIONS = {
        "imagen-3.0-capability-001",  # Imagen3
        "imagen-3.0-generate-002",  # Imagen3 -- currently not allowed for our account )):
        "imagen-3.0-generate-001",  # Imagen3
        "imagegeneration@006",      # Imagen2
        "imagegeneration@002",      # Imagen
    }

    RESOLUTIONS = {
        "1:1": (1024, 1024),
        "3:4": (896, 1280),
        "4:3": (1280, 896),
        "9:16": (768, 1408),
        "16:9": (1408, 768),
    }

    def __init__(
        self,
        project,
        location="us-central1",
        model="imagen-3.0-capability-001",
        verbose=False,
    ):
        """
        Args:
            project (str): Name of the project to use when calling the Gemini client
            location (str): Location to use when calling the Gemini client
            model (str): Gemini model to use. Must be one of self.VERSIONS
        """
        self.project = project
        self.verbose = verbose
        if self.verbose:
            print("="*100)
            print(f"vlm.py line 251: USING PROJECT: {self.project}")
            print("="*100)
        self.location = location
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Imagen3 model")
        self.model = model
        self.client = None

    def __call__(
        self,
        prompt,
        image_path,
        negative_prompt="",
        mask_image_path=None,
        edit_mode="default",
        aspect_ratio="1:1",
        n_images=4,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Imagen3 model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            negative_prompt (str): Text prompt to use for negative prompting
            mask_image_path (None or str): If specified, absolute path corresponding to reference image to use as a
                mask conditioning agent, e.g. for directly supervised image editing
            edit_mode (str): Mode for Imagen3 to operate in, e.g.: "default", "inpainting-remove", ...
            aspect_ratio (str): Aspect ratio to use for generated photos
            n_images (int): Number of images to generate. Can be between 1-4
            seed (int): Random seed to use
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            None or list of Results: Imagen3 raw generated results
        """
        assert_valid_key(key=aspect_ratio, valid_keys=self.RESOLUTIONS, name="Aspect ratio")
        cache = RemoteModelCache.from_env()
        cache_request = {
            "prompt": prompt,
            "image_input": image_digests(image_path),
            "negative_prompt": negative_prompt,
            "mask_image_input": image_digests(mask_image_path),
            "edit_mode": edit_mode,
            "aspect_ratio": aspect_ratio,
            "n_images": n_images,
            "seed": seed,
            "safety_filter_level": "block_few",
            "person_generation": "allow_adult",
        }
        cache_key = cache.key_for(provider="imagen3", model=self.model, request=cache_request)
        if cache.test_enabled:
            return _deserialize_imagen_result(cache.load_response(provider="imagen3", key=cache_key))
        if cache.cache_enabled:
            cached_response = cache.load_response_if_exists(provider="imagen3", key=cache_key)
            if cached_response is not None:
                return _deserialize_imagen_result(cached_response)

        imagen_api = _get_vertex_imagen_api()
        raw_ref_image = imagen_api.RawReferenceImage(
            image=imagen_api.VertexImage.load_from_file(location=image_path),
            reference_id=1,
        )
        ref_images = [raw_ref_image]
        if mask_image_path is not None:
            mask_ref_image = imagen_api.MaskReferenceImage(
                reference_id=1,
                image=imagen_api.VertexImage.load_from_file(location=mask_image_path),
                mask_mode="foreground",
            )
            ref_images.append(mask_ref_image)
        result = None
        if self.client is None:
            imagen_api.vertexai.init(project=self.project, location=self.location)
            self.client = imagen_api.ImageGenerationModel.from_pretrained(self.model)
        for i in range(n_retries):
            if result is not None:
                break
            print(f"Querying Imagen3 [{self.model}]: {i + 1} of {n_retries}...")
            try:
                result = self.client._generate_images(
                    prompt=prompt,
                    edit_mode=edit_mode,
                    reference_images=ref_images,
                    seed=seed,
                    number_of_images=n_images,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_few",
                    person_generation="allow_adult",
                )

            except:
                print(f"Failed attempt {i + 1} of {n_retries}")

        if print_results and result is not None:
            for res in result:
                res._pil_image.show()

        if cache.cache_enabled and result is not None:
            cache.store_response(
                provider="imagen3",
                model=self.model,
                key=cache_key,
                request=cache_request,
                response=_serialize_imagen_result(result),
            )
        return result

    def get_result_images(self, result):
        return [res._pil_image for res in result]



class GPT(VLM_API):
    """
    Class for interfacing with supported GPT models
    """
    VERSIONS = {
        "gpt-image-1",
    }

    IMAGE_SHAPES = {
        "portrait": "1024x1536",
        "square": "1024x1024",
        "landscape": "1536x1024",
    }

    def __init__(
        self,
        model="gpt-image-1",
        api_key=None,
    ):
        """
        Args:
            model (str): GPT model to use. Must be one of self.VERSIONS
            api_key (None or str): OpenAI API key to use. If not set, the OpenAI client falls back
                to the OPENAI_API_KEY environment variable
        """
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="GPT model")
        self.model = model
        self.api_key = api_key

        self.client = None

    def __call__(
        self,
        prompt,
        image_path=None,
        n_images=1,
        n_retries=3,
        image_shape="square",
        print_results=False,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            n_images (int): Number of images to generate
            n_retries (int): Number of retries to attempt
            image_shape (str): Shape of the images to generate. Valid options: {portrait, square, landscape}
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            dict: Output of the model. Valid keys are {"image", "text"} based on the desired model used
        """
        if self.model == "gpt-image-1":
            assert_valid_key(key=image_shape, valid_keys=self.IMAGE_SHAPES, name="image shape")
            cache = RemoteModelCache.from_env()
            cache_request = {
                "prompt": prompt,
                "image_input": image_digests(image_path),
                "n_images": n_images,
                "image_shape": image_shape,
                "size": self.IMAGE_SHAPES[image_shape],
                "quality": "high",
                "input_fidelity": "high",
                "output_format": "png",
                "background": "auto",
            }
            cache_key = cache.key_for(provider="gpt-image", model=self.model, request=cache_request)
            if cache.test_enabled:
                return _deserialize_gpt_image_result(cache.load_response(provider="gpt-image", key=cache_key))
            if cache.cache_enabled:
                cached_response = cache.load_response_if_exists(provider="gpt-image", key=cache_key)
                if cached_response is not None:
                    return _deserialize_gpt_image_result(cached_response)

            result = None
            if self.client is None:
                self.client = OpenAI(api_key=self.api_key)
            for i in range(n_retries):
                if result is not None:
                    break
                print(f"Querying GPT [{self.model}]: {i + 1} of {n_retries}...")
                try:
                    with open(image_path, "rb") as image_file:
                        result = self.client.images.edit(
                            model=self.model,
                            image=image_file,
                            prompt=prompt,
                            n=n_images,
                            size=self.IMAGE_SHAPES[image_shape],
                            quality="high",
                            input_fidelity="high",
                            output_format="png",
                            background="auto",
                            # moderation="auto",
                        )
                except:
                    print(f"Failed attempt {i + 1} of {n_retries}")

            if print_results and result is not None:
                for dat in result.data:
                    image_base64 = dat.b64_json
                    image_bytes = base64.b64decode(image_base64)
                    PILImage.open(BytesIO(image_bytes)).show()

            if cache.cache_enabled and result is not None:
                cache.store_response(
                    provider="gpt-image",
                    model=self.model,
                    key=cache_key,
                    request=cache_request,
                    response=_serialize_gpt_image_result(result),
                )
            return result

        else:
            raise ValueError(f"Got invalid GPT model for inference: {self.model}")

    def get_result_text(self, result):
        # return "".join(res.text for res in result)
        raise NotImplementedError

    def get_result_images(self, result):
        return [PILImage.open(BytesIO(base64.b64decode(dat.b64_json))) for dat in result.data]


class FLUX1(VLM_API):
    """
    Class for interfacing with Flux Kontext model
    """
    VERSIONS = {
        "FLUX.1-Kontext": FluxPipeline,
    }

    IMAGE_SHAPES = {res for res in PREFERRED_KONTEXT_RESOLUTIONS}

    def __init__(
        self,
        model="FLUX.1-Kontext",
        dtype=torch.bfloat16,
        device="cuda",
        enable_cpu_offload=True,
    ):
        """
        Args:
            model (str): FLUX.1 model to use. Must be one of self.VERSIONS
            dtype (torch.dtype): Torch datatype to use
            device (str): Device to use
            enable_cpu_offload (bool): Whether to enable CPU offloading to save on VRAM usage
        """
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="FLUX.1 model")
        self.pipeline = self.VERSIONS[model].from_pretrained(f"black-forest-labs/{model}", torch_dtype=dtype)
        if enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        self.device = device
        # self.pipeline.to(device=self.device)

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        image_path=None,
        guidance_scale=2.5,
        num_inference_steps=28,
        max_sequence_length=512,
        seed=None,
        print_results=False,
    ):
        """
        Calls the FLUX.1 model

        Args:
            prompt (str): Text prompt to use
            negative_prompt (None or str): Optional negative prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            guidance_scale (float): Guidance scale to use during diffusion
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            PIL.Image: Outputted image from the model
        """
        input_img = load_image(image_path)
        assert_valid_key((input_img.width, input_img.height), valid_keys=self.IMAGE_SHAPES, name="input image shape")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        image = self.pipeline(
            image=input_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=input_img.width,
            height=input_img.height,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )[0][0]

        if print_results:
            image.show()

        return image
