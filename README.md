# Read ME
This algorithm was created for a master's thesis.    
It can create crochet patterns with single crochet, increases and decreses based on 3D models.

## Requirements
You need to install *lib igl* python bindings: 
```python
python -m pip install libigl
```
or see https://github.com/libigl/libigl-python-bindings for more details

and install *vedo* specifiaclly pygeodesic
```python
pip install pygeodesic
```
or see https://github.com/mhogg/pygeodesic for more details    

It might also be good to have a library for 3D visualisation. Matplotlib returned the 3D models a little distorted so I used meshplot
```python
conda install -c conda-forge meshplot 
```
or see https://skoch9.github.io/meshplot/tutorial/ for more details

## How to use:

This generator generated patterns for **closed** and **one-edged, non-closed** surfaces. That is patterns for Amigurumis like spheres, cylinders or teardrop shape but also non-closed shapes like vases or pots.    
The 3D model has to fullfill some requirements:     
- It has to be of the file type **OBJ** or **OFF**
- it has to have a vertice at the point you want the pattern to start
- It has to be closed or one-edged otherwise the pattern is not correctly generated

```python
import mesh_to_instructions
import print_pattern
```

You can create a pattern by running mesh_to_instructions.run the function requires a file of a 3D model, the desired stitchwidth (= metric hook size) and a start index as an integer that exists in the file.

```python
instructions, all_points, sample_points, faces, row_edges, column_edges, g_v, g_e, isolines = mesh_to_instructions.run(file, stitch_width , start_index) # includes results for debugging and representation such a edges
#or run 
instructions, sample_points = mesh_to_instructions.create_crochet_pattern(file, stitch_width , start_index)
```

The pattern can be saved as a PDF with  print_pattern.create_pattern_file. This file will include a key for crochet terminology

```python
print_pattern.create_pattern_file(name-of_pattern, goal_filepath, filepath_for_picture, instructions, sample_points, stitch_width)
```

The edges of the crochet graph that will be created in between can be shown in 3D figures.

```python
import meshplot as mp

p = mp.plot(all_points, c = result[1][:,1], shading= {"point_size":2 , "point_color":"red" })

p.add_points(sample_points[0], shading= {"point_color": "red", "point_size":10 } )
p.add_points([len(sample_points)-1], shading= {"point_color": "blue", "point_size":10 } )
color = ["red", "green"]

for key in range(0, len(column_edges)):

    p.add_edges(all_points, column_edges[key], shading = {"line_color": "blue" })#
    p.add_edges(all_points, column_edges[key], shading = {"line_color":color[key%2] , "line_width": 5.0})

```
<img title="Cylinder Crochet Graph" alt="Cylinder crochet graph" src="/images/cylinder.png">

Now generate some patterns and crochet.
