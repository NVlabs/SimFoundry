# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: reorganize prompts into different classes / modules

import json

def prompt_object_setlist():
    return "You are an expert in image captioning.\n\n" + \
    "### Task Description ###\n\n" + \
    "The user will give you an image, and please provide a list of all objects ([object1, object2, ...]) visible in the image. " + \
    "For objects with visible and openable doors and drawers, please also return the number of doors (those with revolute joint, that can rotate around an axis) and drawers (those with prismatic joint, that can slide along a direction).\n\n" + \
    "### Special Requirements ###\n\n" + \
    "1. Treat items that are composed of multiple rigid parts as a single entity; avoid breaking down objects into their components. For instance, mention a wardrobe as one object instead of listing its doors and handles as separate items; " + \
    "2. However, treat items that can be decomposed into completely separate components as individual, separate entities. For instance mention a pot plant/flower as separate objects decomposed into its vase and individual flowers.\n\n" + \
    "3. When captioning, please do not include walls, floors, windows and any items hung from the ceiling in your answer, but please include objects installed or hung on walls.\n\n" + \
    "4. When captioning, please do not include the main surface that the majority of the objects are on. This includes common surfaces such as tables, countertops, etc.\n\n" + \
    "5. When captioning, you can use broader categories. For instance, you can simply specify 'table' instead of 'short coffee table'.\n\n" + \
    "6. Please caption all objects, even if some objects are closely placed, or an object is on top of another, or some objects are small compared to other objects. " + \
    "However, don't come up with objects not in the image or objects that are partially clipped towards the edge of the image.\n\n" + \
    "7. Please do not add 's' or 'es' suffices to countable nouns. For example, you should caption multiple apples as 'apple', not 'apples'.\n\n" + \
    "8. When counting the number of doors and drawer, pay attention to the following:\n\n" + \
    "(1). A child link cannot be a door and a drawer at the same time. When you are not sure if a child link is a door or a drawer, choose the most likely one.\n" + \
    "(2). Please only count openable doors and drawers. Don't include objects with fixed and non-openable drawers/shelves/baskets (e.g., shelves/baskets/statis drawers of bookshelves, shelves, storage carts). For these objects, just give me the caption (e.g., bookshelf, shelf, storage cart).\n\n" + \
    "Example output1: [banana, cabinet(3 doors & 3 drawers), chair]\n" + \
    "Example output2: [wardrobe(2 doors), table, storage cart]\n" + \
    "Example output3: [television, apple, shelf]\n" + \
    "Example output4: [cabinet(8 drawers), desk, frying pan]\n\n\n"

def prompt_object_setlist_specific_tabletop():
    return "You are an expert in image captioning.\n\n" + \
    "### Task Description ###\n\n" + \
    "The user will give you an image, and please provide a list of all objects ([object1, object2, ...]) visible in the image on the main flat surface (such as the floor, or desk, or table) in the center of the image. " + \
    "1. Treat items that can be decomposed into completely separate components as individual, separate entities. For instance mention a pot plant/flower as separate objects decomposed into its vase and individual flowers.\n\n" + \
    "2. When captioning, please do not include walls, floors, windows and any items hung from the ceiling in your answer, but please only include objects that are on top of the main surface.\n\n" + \
    "3. When captioning, please do not include the main surface that the majority of the objects are on. This includes common surfaces such as desks, tables, countertops, etc.\n\n" + \
    "4. When captioning, be concise but specific, including key distinguishing characteristics when necessary. At least one descriptive adjective should be used. For instance, it is not acceptable to simply specify 'banana'; conversely, it is usually unnecessary to use the overly verbose description 'small banana with brown spots'. Instead, a good example would be 'yellow banana', or 'ripe banana'.\n\n" + \
    "5. If an object has an obvious brand that is seen, please specify it in its description. For example, the caption 'apple iphone' is preferred over the more generic 'cell phone'.\n\n" + \
    "6. Please caption all objects, even if some objects are closely placed, or an object is on top of another, or some objects are small compared to other objects. " + \
    "However, don't come up with objects not in the image or objects that are partially clipped towards the edge of the image.\n\n" + \
    "7. If there are multiple identical objects, only mention a single individual instance. For example, if there are three black pens, only mention 'black pen'.\n\n" + \
    "8. Please use chain-of-thought reasoning before arriving to your final output. Example final outputs are given below:\n\n" + \
    "Example output1: [ripe banana, car keys, kids toothpaste]\n" + \
    "Example output2: [red pen, black pen, black notebook, square stickypad]\n" + \
    "Example output3: [samsung television, green book, blue book]\n" + \
    "Example output4: [cuisinart frying pan, wooden fork, metal knife, plastic spoon, cloth napkin]\n\n\n"

def prompt_object_setlist_specific_floorplane(floor_category):
    return f"You are an expert in image captioning.\n\n" + \
    "### Task Description ###\n\n" + \
    f"The user will give you an image, and please provide a list of all objects ([object1, object2, ...]) visible in the image on the {floor_category} plane. " + \
    "1. Treat items that can be decomposed into completely separate components as individual, separate entities. For instance mention a pot plant/flower as separate objects decomposed into its vase and individual flowers.\n\n" + \
    f"2. When captioning, please do not include walls, floors, windows and any items hung from the ceiling in your answer, but please only include objects that are on top of the main surface, which is the {floor_category}.\n\n" + \
    f"3. When captioning, please do not include the main surface that the majority of the objects are on. That is, do not include {floor_category} in your answer.\n\n" + \
    "3.1 When captioning, do not include items that are not solid or that are stuck onto other objects such as magnets, sticky notes, tape, etc.\n\n" + \
    "4. When captioning, be concise but specific, including key distinguishing characteristics when necessary. At least one descriptive adjective should be used. For instance, it is not acceptable to simply specify 'banana'; conversely, it is usually unnecessary to use the overly verbose description 'small banana with brown spots'. Instead, a good example would be 'yellow banana', or 'ripe banana'.\n\n" + \
    "5. If an object has an obvious brand that is seen, please specify it in its description. For example, the caption 'apple iphone' is preferred over the more generic 'cell phone'.\n\n" + \
    "6. For objects that represent a single semantic entity but are composed of multiple individual sub-objects, please specify the main object in an unambiguous way. For example, grapes on a vine should be specified as 'green grape bunch' instead of simply 'green grapes'.\n\n" + \
    "7. Please caption all objects, even if some objects are closely placed, or an object is on top of another, or some objects are small compared to other objects. " + \
    "However, don't come up with objects not in the image or objects that are mostly clipped towards the edge of the image. If more than 75\% of the object is visible, include it.\n\n" + \
    "8. If there are multiple identical objects, only mention a single individual instance. For example, if there are three black pens, only mention 'black pen'.\n\n" + \
    "9. Order the objects from least occluded to most occluded. For example, if a banana is in front of a plate, then the banana should be mentioned first, and the plate should be mentioned second.\n\n" + \
    "10. Please use chain-of-thought reasoning before arriving to your final output.\n\n" + \
    "Return your final answer as a JSON object in the following format: " + \
    '''json
    {{
        "objects": ["object1", "object2", ...],
        "reasoning": "your reasoning here"
    }}
    ''' + \
    "Example output1: " + \
    '''json
    {{
        "objects": ["ripe banana", "car keys", "kids toothpaste"],
        "reasoning": "the user asked for objects on the floor plane, so we only include objects that are on the floor plane"
    }}
    ''' + \
    "Example output2: " + \
    '''json
    {{
        "objects": ["red pen", "black pen", "black notebook", "square stickypad"],
        "reasoning": "the user asked for objects on the floor plane, so we only include objects that are on the floor plane"
    }}
    '''

def prompt_object_setlist_specific_floorplane_v2(floor_category):
    return f"You are helping an image segmentation system.\n\n" + \
    "### Task Description ###\n\n" + \
    f"The user will give you an image. Please provide a list of all visible objects on the {floor_category} plane. " + \
    "For each object, return exactly three short names in the format: primary name / alias 1 / alias 2.\n\n" + \
    "Rules:\n\n" + \
    "0. The first name is the most important. It must be the shortest, most common, and most reliable name for segmentation.\n\n" + \
    "1. The first name should usually be 1 to 3 words.\n\n" + \
    "2. Prefer simple visually grounded names, usually in the form 'color + object noun', such as 'blue marker', 'black eraser', or 'yellow bowl'.\n\n" + \
    "3. If the object is a toy version (not a real object), the object name MUST include the keyword 'toy'. Example: use 'toy green pear' (not 'green pear'). Apply this to the primary name and keep aliases consistent.\n\n" + \
    "4. If an object's color is similar to the background or support surface (low contrast), add one short but strong visual detail in the PRIMARY name to improve segmentation reliability. Good details include shape, part, material, or pattern, such as 'white mug with handle', 'beige matte bowl', 'clear ribbed bottle', or 'black remote with buttons'.\n\n" + \
    "5. The second and third names should be close synonyms of the first name, not broader or more abstract descriptions.\n\n" + \
    "6. Avoid abstract or functional phrases such as 'writing instrument', 'desk accessory', or 'food container'.\n\n" + \
    "7. Avoid unnecessary adjectives such as 'fresh', 'ripe', 'large', 'dark', or 'small' unless they are necessary to distinguish the object.\n\n" + \
    "8. If a brand is visible, only include it if it is the most recognizable and most useful segmentation name.\n\n" + \
    "9. Default to grouping parts that usually move together as a single segment target. Examples: flowers with vase, tape with dispenser, toothbrush in a holder cup when inserted and moving together as placed.\n\n" + \
    "10. Only split into separate objects when components are clearly independent and not physically coupled (e.g., a banana resting on a plate, a pen lying on a notebook).\n\n" + \
    "11. Do not include walls, floors, windows, ceiling objects, or the main supporting surface itself.\n\n" + \
    f"12. Do not include the main surface itself, which is the {floor_category}.\n\n" + \
    "13. If multiple identical objects exist, only list one category entry for that object type.\n\n" + \
    "14. Include small objects if they are clearly visible, but do not invent objects or include objects that are mostly clipped out of frame.\n\n" + \
    "Return your final answer as a JSON object in the following format: " + \
    '''json
    {{
        "objects": [
            "blue marker / blue dry-erase marker / whiteboard marker",
            "black eraser / whiteboard eraser / rectangular eraser"
        ]
    }}
    ''' + \
    "Example output1: " + \
    '''json
    {{
        "objects": ["vase with flowers / flower arrangement / flower vase", "car keys / keyring / keys", "toothpaste tube / kids toothpaste / toothpaste"]
    }}
    ''' + \
    "Example output2: " + \
    '''json
    {{
        "objects": ["red marker / red dry-erase marker / marker", "black notebook / notebook / dark notebook", "square sticky note / sticky note / note pad"]
    }}
    '''
    
# generate bbox from Gemini
def prompt_object_setlist_specific_floorplane_v3(floor_category):
    return f"You are helping an image segmentation system.\n\n" + \
    "### Task Description ###\n\n" + \
    f"The user will give you an image. Please provide a list of all visible objects on the {floor_category} plane. " + \
    "For each object, return exactly three short names in the format: primary name / alias 1 / alias 2, and also return one bounding box.\n\n" + \
    "Rules:\n\n" + \
    "0. The first name is the most important. It must be the shortest, most common, and most reliable name for segmentation.\n\n" + \
    "1. The first name should usually be 1 to 3 words.\n\n" + \
    "2. Prefer simple visually grounded names, usually in the form 'color + object noun', such as 'blue marker', 'black eraser', or 'yellow bowl'.\n\n" + \
    "3. If an object's color is similar to the background or support surface (low contrast), add one short but strong visual detail in the PRIMARY name to improve segmentation reliability. Good details include shape, part, material, or pattern, such as 'white mug with handle', 'beige matte bowl', 'clear ribbed bottle', or 'black remote with buttons'.\n\n" + \
    "4. The second and third names should be close synonyms of the first name, not broader or more abstract descriptions.\n\n" + \
    "5. Avoid abstract or functional phrases such as 'writing instrument', 'desk accessory', or 'food container'.\n\n" + \
    "6. Avoid unnecessary adjectives such as 'fresh', 'ripe', 'large', 'dark', or 'small' unless they are necessary to distinguish the object.\n\n" + \
    "7. If a brand is visible, only include it if it is the most recognizable and most useful segmentation name.\n\n" + \
    "8. Default to grouping parts that usually move together as a single segment target. Examples: flowers with vase, tape with dispenser, toothbrush in a holder cup when inserted and moving together as placed.\n\n" + \
    "9. Only split into separate objects when components are clearly independent and not physically coupled (e.g., a banana resting on a plate, a pen lying on a notebook).\n\n" + \
    "10. Do not include walls, floors, windows, ceiling objects, or the main supporting surface itself.\n\n" + \
    "10.b. Do not include items that are not solid or that are stuck onto other objects such as magnets, sticky notes, tape, etc.\n\n" + \
    f"11. Do not include the main surface itself, which is the {floor_category}.\n\n" + \
    "12. If multiple identical objects exist, only list one category entry for that object type.\n\n" + \
    "13. Include small objects if they are clearly visible, but do not invent objects or include objects that are mostly clipped out of frame.\n\n" + \
    "14. Return one bounding box for every object as `box_2d` in [ymin, xmin, ymax, xmax] format.\n\n" + \
    "15. `box_2d` must be normalized to 0-1000 integer coordinates, where 0 is top/left and 1000 is bottom/right of the image.\n\n" + \
    "16. Bounding boxes should tightly cover the visible object area and stay within image bounds.\n\n" + \
    "17. Keep the same object order between `objects` and `detections`.\n\n" + \
    "18. In each `detections` item, `name` must exactly match the corresponding string in `objects`.\n\n" + \
    "19. If no valid objects are visible on the surface (after applying all exclusion rules above), " + \
    "return empty arrays for both `objects` and `detections`. Do not invent placeholder names, " + \
    "sentinel strings, or explanatory text in place of objects.\n\n" + \
    "Return your final answer as a JSON object in the following format: " + \
    '''json
    {{
        "objects": [
            "blue marker / blue dry-erase marker / whiteboard marker",
            "black eraser / whiteboard eraser / rectangular eraser"
        ],
        "detections": [
            {{
                "name": "blue marker / blue dry-erase marker / whiteboard marker",
                "box_2d": [220, 150, 360, 260]
            }},
            {{
                "name": "black eraser / whiteboard eraser / rectangular eraser",
                "box_2d": [210, 285, 280, 360]
            }}
        ]
    }}
    ''' + \
    "Example output1: " + \
    '''json
    {{
        "objects": [
            "vase with flowers / flower arrangement / flower vase",
            "car keys / keyring / keys",
            "toothpaste tube / kids toothpaste / toothpaste"
        ],
        "detections": [
            {{
                "name": "vase with flowers / flower arrangement / flower vase",
                "box_2d": [130, 110, 720, 360]
            }},
            {{
                "name": "car keys / keyring / keys",
                "box_2d": [610, 380, 760, 560]
            }},
            {{
                "name": "toothpaste tube / kids toothpaste / toothpaste",
                "box_2d": [500, 600, 640, 850]
            }}
        ]
    }}
    ''' + \
    "Example output2: " + \
    '''json
    {{
        "objects": [
            "red marker / red dry-erase marker / marker",
            "black notebook / notebook / dark notebook",
            "square sticky note / sticky note / note pad"
        ],
        "detections": [
            {{
                "name": "red marker / red dry-erase marker / marker",
                "box_2d": [430, 220, 520, 580]
            }},
            {{
                "name": "black notebook / notebook / dark notebook",
                "box_2d": [360, 520, 760, 900]
            }},
            {{
                "name": "square sticky note / sticky note / note pad",
                "box_2d": [420, 900, 560, 990]
            }}
        ]
    }}
    ''' + \
    "Example output3 (no valid objects): " + \
    '''json
    {{
        "objects": [],
        "detections": []
    }}
    '''
   


def prompt_floor_setlist():
    return "You are an expert in image captioning.\n\n" + \
    "### Task Description ###\n\n" + \
    "The user will give you an image, and please provide a list of all objects that could be considered as the floor plane in the scene. " + \
    "Please think through your process first via chain-of-thought. Example final outputs are given below:\n\n" + \
    "1. Each object should be a single entity. For instance, a table is a single entity, not a collection of floor tables.\n\n" + \
    "2. The floor plane should not have any objects below it that are visible. For example, if a table is visible in the image, but the ground below it is also visible, then the table is not the floor plane.\n\n" + \
    "3. Return the objects in order of priority, from the most likely to be the floor plane to the least likely.\n\n" + \
    "4. Please describe each object generally, and in a single word. For example, ['floor', 'carpet', 'mat'] instead of ['tile floor', 'red carpet', 'floor mat'].\n\n" + \
    "5. Return the objects in the format: [object1, object2, ...].\n\n" + \
    "6. Only return the list of objects, no other text or explanation.\n\n" + \
    "Example output1: [floor, carpet]\n" + \
    "Example output2: [floor, table, countertop, desk]\n" + \
    "Example output3: [floor, carpet, mat]\n" + \
    "Example output4: [floor, carpet, mat, desk]\n\n\n"

def prompt_object_removal_selection(obj_list):
    obj_list_str = "\n"
    for i, name in enumerate(obj_list):
        obj_list_str += f"({i + 1}) {name}\n"
    n_objects = len(obj_list)
    return f"""There are {n_objects} objects in the photo, uniquely marked with numbers 1-{n_objects} with a white number within a black circle. Each number corresponds to the following objects:
{obj_list_str}
Please tell me the number corresponding to the object in the image that satisfies the following properties the most, in order of priority:

- not occluded by other objects
- not beneath any other other objects
- towards the foreground of the image

Please think through your process first via chain-of-thought. Then, provide your final answer with the phrase ANSWER: [number]."""


# def prompt_remove_object(category):
#     return f"""A detailed, high-resolution image illustrating the step of removing the {category} marked with a '1' and outlined in a bright green color. The scene is meticulously rendered to reveal the portions of the image obscured by the removed object, ensuring that all other elements and details remain unchanged. The image emphasizes a before-and-after comparison, showcasing the clarity of the newly revealed components of the scene, achieved by digitally eliminating the specified object, without any alteration to the surrounding environment or objects. The lighting and color scheme remain consistent, creating a clear and accurate depiction of the changes resulting from the object removal."""

def prompt_list_articulated_objects(object_list):
    return f"You are an expert in image captioning.\n\n" + \
    "### Task Description ###\n\n" + \
    f"In the following list of objects {object_list}, please list all the objects that are articulated, i.e., have a movable part or joint. " + \
    "You will also be given images of the objects. Analyze the images and determine if the object is articulated. Do not include objects where all articulated parts are smaller than 6 inches in every dimension (length, width, height). " + \
    "Return the articulated objects in the following format:\n\n" + \
    "Return your final answer as a JSON object in the following format: " + \
    '''json
    {{
        "articulated_objects": ["object1", "object2", ...],
        "non_articulated_objects": ["object1", "object2", ...],
    }}
    ''' + \
    "Example output1: " + \
    '''json
    {{
        "articulated_objects": ["cabinet", "fridge", "laptop"],
        "non_articulated_objects": ["table", "shelf", "banana"],
    }}  
    '''
    

def prompt_remove_object(category):
    return f"""A detailed, high-resolution image illustrating the step of removing the {category} marked with a '1' and outlined in a bright green color. The scene is meticulously rendered to reveal the portions of the image obscured by the removed {category}, ensuring that all other elements and details remain unchanged. The image emphasizes a before-and-after comparison, showcasing the clarity of the newly revealed components of the scene, achieved by digitally eliminating the specified {category}, without any alteration to the surrounding environment or objects. The lighting and color scheme remain consistent, creating a clear and accurate depiction of the changes resulting from the {category} removal."""

def prompt_classify_object():
    return f"""Please describe the object you see in the foreground of the image. The description should be a concise phrase and use as few words as possible, while still being clear and not ambiguous. Provide you answer in the format: ANSWER: <description>"""

def prompt_classify_annotated_object(number):
    return f"""Please describe the object you see in the foreground of the image, annotated with the number {number}. The description should be a concise phrase and use as few words as possible, while still being clear and not ambiguous. Provide you answer in the format: ANSWER: <description>"""

def prompt_classify_outlined_object():
    return f"""Please describe the object you see in the foreground of the image that is outlined in bright green. The description should be a concise phrase and use as few words as possible, while still being clear and not ambiguous. Provide your answer in the format: ANSWER: <description>"""

def prompt_classify_number_in_circle(number):
    return f"""Does the image show a number {number} inside a darker colored circle? Please only answer 'yes' or 'no'. Provide your answer in the format: ANSWER: <yes/no>"""

def prompt_upsample_image():
    return f"""Edit this photo to significantly increase the resolution, photorealistic, ultra-detailed, 8K, shot on a Canon EOS R5, F16, 89mm, sharp details. Include the entire object in the image.""" # Do not change any other aspect of the object and make sure the entire object fits in the image frame."""

def prompt_upsample_image_rotate():
    return f"""Edit this photo to significantly increase the resolution, photorealistic, ultra-detailed, 8K, shot on a Canon EOS R5, F16, 89mm, sharp details. If necessary, rotate the camera view so that the isometric view is shown.""" # Do not change any other aspect of the object and make sure the entire object fits in the image frame."""

def prompt_infill_image(obj_name):
    return f"""You are given a photo of a {obj_name}. However, some details of the {obj_name} may be missing. If so, fill in the missing details and parts of the object.
    Preserve the original object's pose, position, and orientation as well as its appearance. Do not change the object's color, material or shape, only fill in the missing details.
    The resulting image should be a complete and accurate representation of the {obj_name}, showing the object's intricate details and surface texture. Ultra-realistic. """

def prompt_infill_image_no_conditioning(obj_name):
    return f"""Fill in the missing details and parts of the object. Preserve the original object's pose, position, and orientation as well as its appearance. Do not change the object's color, material or shape, only fill in the missing details. 
    The resulting image should be a complete and accurate representation of the object, showing the object's intricate details and surface texture. The quality of the image should be the same as the original image."""

def prompt_upsample_image_gemini():
    return f"""A high-resolution, studio-lit product photograph of this object resting stably on the ground, presented on a polished concrete surface. 
    The lighting is a three-point softbox setup designed to create soft, diffused highlights and eliminate harsh shadows. 
    The camera angle is a slightly elevated 45-degree shot to showcase its clean lines. 
    Camera is zoomed in on the object, showing the object's intricate details and surface texture. 
    Ultra-realistic in 4K resolution.
    """
    #Rotate the object so its bottom surface is flat on the ground, with the ground surfacepointing up.


def prompt_check_object_validity(obj_name):
    return f"""Please check if the provided image of the object {obj_name} is valid. The object should be a complete and accurate representation of the {obj_name}, showing the object's intricate details and surface texture.
    Consider if the image is similar to other examples of the object {obj_name} in the real world. The image of the object should be realistic and representative of the object in the real world.
    Please think through your process first via chain-of-thought. Then, provide your final answer in the format:
    '''json
    {{
        "validity": "yes/no",
        "reasoning": "your reasoning here"
    }}
    '''
    """

def prompt_topk_image_select(n_candidates):
    return f"""Above shown are {n_candidates + 1} images. The first image shows the reference image of a real-world object, and the following {n_candidates} images show candidate poses of a 3D object model trying to match the reference image's object pose. Which of the {n_candidates} options are the closest to the pose? Please consider key distinguishing details from the object when determining the correct pose. Please use chain-of-thought reasoning. Please specify your answer as the index of the image that most closely matches the original reference object's pose, i.e.: a number between 2 and {n_candidates + 1} (not including 1 because that is the reference image). Provide your answer in the format: ANSWER: <index>"""

# , making sure they match as much as possible to the reference image

def prompt_canonical_frame_select(n_candidates):
    return f"""Above shown are {n_candidates} images, labelled OPTION 1 through OPTION {n_candidates} in the top-left corner. They are frames of the same real-world scene captured from different camera viewpoints.

Exactly one of them will be used to reconstruct the whole scene in simulation: every object visible on the support surface (table, desk, counter, floor) will be cropped out of that single frame and turned into a 3D mesh, and that frame's camera will define the world frame.

Pick the option that would reconstruct best, judging in this order of priority:

1. Every object resting on the support surface is fully visible, and none is hidden behind or underneath another object.
2. The objects occupy as many pixels as possible -- prefer closer viewpoints, because a small object rendered at a low pixel count reconstructs into a poor mesh. Pay particular attention to the SMALLEST object in the scene.
3. The image is sharp and in focus, with no motion blur, especially on the objects.
4. No object is clipped by the edge of the frame.
5. The support surface is clearly visible, with a large part of its extent in view.

Please think through your process first via chain-of-thought, explicitly comparing how large and how occluded the smallest object is in each option. Then provide your final answer as the option number, in the format: ANSWER: <option number between 1 and {n_candidates}>"""


def prompt_object_mass_friction(obj_phrase, bounding_box_cm, volume_cm):
    bbox_str = f" x ".join([f"{val:.2f}cm" for val in bounding_box_cm])
    return f"""Shown is a picture of a {obj_phrase}. Please give the estimated mass (kg) and friction (unitless) of the shown object, given that its bounding box dimensions are {bbox_str} and corresponding volume is {volume_cm:.2f}cm^3. Use chain of thought to think through the predicted material and physical properties of the shown object before deciding on the mass and friction. 
    Answer should be in the following format: json
    {{
        "mass": <value>,
        "friction": <value>
    }}
    """


def prompt_articulated_object_parts_properties(
    obj_phrase: str,
    parts_info: list[dict],  # [{"name": "door_left", "bounding_box_cm": [...], "volume_cm": ...}, ...]
):
    """
    Prompt VLM to estimate physical properties for each part of an articulated object.
    """
    parts_desc = "\n".join([
        f"  - {p['name']}: bounding box {p['bounding_box_cm'][0]:.2f}cm x {p['bounding_box_cm'][1]:.2f}cm x {p['bounding_box_cm'][2]:.2f}cm, volume {p['volume_cm']:.2f}cm³"
        for p in parts_info
    ])
    
    return f"""Shown is a picture of a {obj_phrase}, which is an articulated object with the following parts:
{parts_desc}

For each part, estimate:
1. Mass (kg) - based on typical material for that part type
2. Friction coefficient (unitless) - surface property
3. Joint damping (if movable) - how much resistance the joint has

Consider that different parts likely have different materials:
- A cabinet body is typically wood/particleboard
- Doors/drawers share the body material but may have different handles (metal)
- Handles and knobs are often metal or plastic

Use chain-of-thought reasoning for each part. Return in JSON format:
{{
  "parts": [
    {{"name": "part_name", "mass_kg": 0.0, "friction": 0.0, "joint_damping": 0.0}},
    ...
  ],
  "reasoning": "your reasoning here"
}}
"""

def prompt_check_phrase_removed(obj_phrase):
    return f"""Shown is a picture containing a scene of objects. Does the scene include (show) a '{obj_phrase}'? Please only answer 'yes' or 'no'. Provide your answer in the format: ANSWER: <yes/no>"""


def prompt_occlusion_ordering(obj_list):
    obj_list_str = "\n"
    for i, name in enumerate(obj_list):
        obj_list_str += f"({i + 1}) {name}\n"
    n_objects = len(obj_list)
    return f"""There are {n_objects} objects in the photo, uniquely marked with numbers 1-{n_objects} with a white number within a black circle. Each number corresponds to the following objects:
{obj_list_str}
For each object (represented by its number), please list the other object(s) (represented by their corresponding numbers) that are behind the given object. An object is considered behind it if the given object is explicitly blocking part of the other object within the image. Please think through your process first via chain-of-thought.

For a scene that has 5 numbered objects, an example answer should look something like this:

ANSWER:

- Object 1 is in front of objects 3,5
- Object 2 is in front of no other objects
- Object 3 is in front of objects 2
- Object 4 is in front of objects 1,2,3
- Object 5 is in front of objects 2,3

"""

# def prompt_flux_object_removal(obj_phrase):
#     return f"""Remove the {obj_phrase} outlined in the bright green box, showing all parts of the scene occluded behind and below it. The {obj_phrase} should not be visible. Keep all other aspects of the scene unchanged. Maintain the same lighting, pose, and position of all other objects in the scene, and maintain the object spatial layout composition. Do not remove any object that is not the {obj_phrase}."""

# def prompt_flux_object_removal(obj_phrase):
#     return f"""Remove the {obj_phrase} outlined in the bright green box, showing all parts of the scene occluded behind and below it. The {obj_phrase} should not be visible. Keep all other aspects of the scene unchanged. Maintain the same lighting, pose, and position of all other objects in the scene. Do not remove any object that is not the {obj_phrase}."""

def prompt_flux_object_removal_bbox(obj_phrase):
    return f"""Remove the {obj_phrase} outlined in the bright green box, showing all parts of the scene occluded behind and below it. The {obj_phrase} should not be visible. Keep all other aspects of the scene unchanged. Maintain the same lighting, pose, and position of all other objects in the scene. Do not remove any object that is not the {obj_phrase}."""

def prompt_flux_object_removal_outline(obj_phrase):
    return f"""Remove the {obj_phrase} outlined in bright green, showing all parts of the scene occluded behind and below it. The {obj_phrase} should not be visible. Keep all other aspects of the scene unchanged. Maintain the same lighting, pose, and position of all other objects in the scene. Do not remove any object outside of the bright green boundary."""

def prompt_flux_object_removal_mask(obj_phrase):
    return f"""Replace the bright green pixels with empty air. Maintain the same lighting, pose, and position of all objects in the scene."""

# def prompt_flux_object_removal(obj_phrase):
#     return f"""Remove the {obj_phrase} outlined in the bright green box, showing all parts of the scene occluded behind and below it. The {obj_phrase} should not be visible. Keep all other aspects of the scene unchanged. Maintain the same lighting, pose, and position of all other objects in the scene, and maintain the object spatial layout composition. Maintain the background behind the {obj_phrase}. Do not remove any object that is not the {obj_phrase}."""

# def prompt_flux_object_removal(obj_phrase):
#     return f"""Remove the {obj_phrase} outlined in the bright green box, showing all parts of the scene occluded behind and below it. The {obj_phrase} should not be visible. Keep all other aspects of the scene unchanged. Maintain the same lighting, pose, and position of all other objects in the scene, and maintain the object spatial layout composition."""

def prompt_flux_object_completion(obj_phrase):
    return f"""Shown is an occluded image of a {obj_phrase}. Complete the missing parts of this {obj_phrase} so the object geometry is complete and fill in the remaining background with a plain white background."""

def prompt_flux_object_completion_and_upsample(obj_phrase):
    # Can try "a" instead of "this"
    return f"""Show a single {obj_phrase} in the foreground and a natural surface in the background."""

def prompt_flux_object_completion_and_upsample_preserve(obj_phrase):
    return f"""Show a single {obj_phrase} in the foreground and a natural flat surface in the background. Preserve the {obj_phrase} visual appearance, but fix the occluded parts. Ultra-detailed, 8K, shot on a Canon EOS R5, F16, 89mm, sharp details."""

# Show a single yellow and black screwdriver in the foreground and a natural surface in the background. Preserve the yellow and black screwdriver visual appearance, but fix the occluded parts.


def parse_json_response(response: str) -> dict:
    if response is None:
        raise ValueError("Expected JSON response text, got None")

    text = response.strip()
    decoder = json.JSONDecoder()

    start_index = text.find('```json')
    if start_index != -1:
        end_index = text.find('```', start_index + 7)
        if end_index != -1:
            json_str = text[start_index + 7:end_index].strip()
            return json.loads(json_str)

    stripped = text.strip('`').strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            return parsed
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Could not find a valid JSON object or array", text, 0)
