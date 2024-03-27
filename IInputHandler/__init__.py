'''
===============================================================================
-- IInputHandler --------------------------------------------------------------
===============================================================================
Manages preprocessing requirements for different pipelines, such as resizing, 
normalizing, and generating multi-view images.

    TripoSR:
    - Objective: To standardize the input for the model by resizing and 
        normalizing the input image.
    - Input: A single 2d RGB image.
    - Output: A 2d masked image, resized to a specific resolution.

    CRM:
    - Objective: To generate six orthographic images from a single input image, 
        providing a comprehensive view necessary for reconstructing the 3D 
        geometry.
    - Input: A single 2d RGB image.
    - Output: Six orthographic 2d masked images resized to a specific 
        resolution.

'''