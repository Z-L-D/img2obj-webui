'''
===============================================================================
-- IReconstructor -------------------------------------------------------------
===============================================================================
Encapsulates the functionality for encoding images into a representation suited 
for 3D reconstruction

    TripoSR:
    - Objective: transform the triplane representation into a detailed 3D 
        textured mesh using Torchmcubes.
    - Input: Latent vectors from the Image Encoding stage.
    - Output: Triplane representation of the 3D object.

    CRM:
    - Objective:  transform the triplane representation into a detailed 3D 
        textured mesh using Flexicubes
    - Input: Triplane representation of the 3D object.
    - Output: A detailed and textured 3D mesh of the object.

'''