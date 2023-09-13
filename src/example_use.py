import mesh_to_instructions
import print_pattern

#################################################################
####### Example usage of algorithm
##################################################################

stitch_width = 5
name = 'Cylinder'
model_file = '../3Dmodels/cylinder2.obj'
target_path = './exmples/Exaple_cylinder.pdf'
image_path = '../images/cylinder.png'

instructions, sample_points = mesh_to_instructions.create_crochet_pattern(model_file, stitch_width)
    
print_pattern.create_pattern_file(name,target_path, image_path, instructions, sample_points,stitch_width)