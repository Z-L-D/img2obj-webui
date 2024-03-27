'''
===============================================================================
-- IEncoder -------------------------------------------------------------------
===============================================================================
Encapsulates the functionality for encoding images into a representation suited 
for 3D reconstruction

    TripoSR:
    - Objective: To encode the preprocessed image into a set of latent vectors 
        that capture its essential features using DINOv1.
    - Input: A 2d masked image, resized to a specific resolution.
    - Output: A set of latent vectors encoding global and local features of the 
        image.

    CRM:
    - Objective: To encode the preprocessed images into Canonical Coordinate 
        Maps (CCM).
    - Input: Six orthographic 2d masked images resized to a specific 
        resolution.
    - Output: Six orthographic 2d masked images and six Canonical Coordinate 
        Maps.

'''