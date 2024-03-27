'''
===============================================================================
-- IDecoder -------------------------------------------------------------------
===============================================================================
Encapsulates the functionality for encoding images into a representation suited 
for 3D reconstruction

    TripoSR:
    - Objective: To transform the latent vectors into a triplane representation
    - Input: Latent vectors from the Image Encoding stage.
    - Output: Triplane representation of the 3D object.

    CRM:
    - Objective: To encode the orthographic images and Canonical Coordinate Maps 
        (CCMs) into a triplane representation
    - Input: Six orthographic 2d masked images and six Canonical Coordinate 
        Maps.
    - Output: Triplane representation of the 3D object.

'''