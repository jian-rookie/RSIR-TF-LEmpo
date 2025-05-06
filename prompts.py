airplane_llama_prompt = '''
I have a remote sensing image about airplane. Given the attribute of the image and a modification caption to edit the image, please generate the description of the edited image based on the image attribute and modification caption. 
The image attribute I provide will begin with "Image Attribute:". The modified caption I provide will begin with “Modification caption:". The edited attribute you generate should begin with “Edited Description:". The description you generate should only talk about the edited image and not the original image. 
The description should be concise and informative and contain all attributes. Avoid adding imaginary things. Use the examples below for reference.

Image Attribute: {"airplane": {"number": 4, "distance": "sparse", "color": "white", "wings": "white", "tailpiece": "white", "direction": "bottom left or bottom right or left"}, "airport": "parking place"}
Modification caption: more lawn on two sides and less airplanes number and change airplanes color from white to gray and change airplanes direction from bottom left or bottom right or left to bottom right
Edited Description: no more than four airplanes with gray color facing the bottom right park at airport and more lawn on two sides

Image Attribute: {"airplane": {"number": 1, "color": "white", "wings": "gray", "tailpiece": "gray", "direction": "bottom right"}, "cars": "two sides", "containers": "two sides", "airport": "parking place"}
Modification caption: change airplanes direction from bottom right to left and less cars on two sides and less containers on two sides
Edited Description: one white airplane with gray wings and tailpiece facing the left parks at airport

Image Attribute: {"airplane": {"number": 3, "distance": "sparse", "color": "white", "wings": "white", "tailpiece": "white", "direction": "left or right"}, "airport": "parking place"}
Modification caption: change parking place from airport to lawn and bring airplanes closer together and change airplanes direction from left or right to bottom left
Edited Description: three white airplanes with white wings and tailpiece facing the bottom left park on the lawn in tight distance

Image Attribute: {"airplane": {"number": 1, "color": "purple", "wings": "white", "tailpiece": "gold and red and white", "direction": "bottom right"}, "containers": "one side", "airport": "parking place"}
Modification caption: change airplanes wings from white to gray and change airplanes tailpiece from gold and red and white to gray and red and yellow
Edited Description: one purple airplane with gray wings and gray and red and yellow tailpiece facing the bottom right parks at airport with containers on one side

Image Attribute: {"airplane": {"number": 1, "color": "white", "wings": "gray", "tailpiece": "white", "direction": "top"}, "containers": "two sides", "buildings": {"number": 1, "color": "gray", "position": "one side"}, "airport": "parking place"}
Modification caption: less buildings number and less buildings position on one side and less buildings color on gray and change airplanes direction from top to bottom
Edited Description: one white airplane with gray wings facing the bottom parks at airport with containers on two sides
'''

airplane_llama_template = '''
Image Attribue: {}
Modification caption: {}
'''

airplane_convert_prompt = '''
I have a remote sensing image about airplane. Given the image attribute, please generate a description of the image based on the image attribute.
The image attribute I provide will begin with "Image Attribute:". The description you generate should begin with “Description:". 
The description should be concise and informative and contain all attributes. Avoid adding imaginary things. Use the examples below for reference.

Image Attribute: {"airplane": {"number": 1, "color": "white", "wings": "gray", "tailpiece": "gray", "direction": "top left"}, "cars": "one side", "buildings": {"number": 1, "color": "gray", "position": "one side"}, "airport": "parking place"}
Description: one white airplane with gray wings and tailpiece facing the top left parks at the airport, with cars on one side, a gray building on one side

Image Attribute: {"airplane": {"number": 2, "distance": "tight", "color": "white", "wings": "gray", "tailpiece": "gray", "direction": "left"}, "roads": "one side", "airport": "parking place"}
Description: two white airplanes with gray wings and tailpiece facing the left park at the airport in tight distance, with roads on one side

Image Attribute: {"airplane": {"number": 2, "distance": "tight", "color": "white", "wings": "white", "tailpiece": "white", "direction": "bottom or top"}, "dark airport": "parking place"}
Description: two white airplanes with white wings and tailpiece facing the bottom or top park at the dark airport in tight distance
'''

airplane_convert_template = '''
Image Attribute: {}
'''

tennis_convert_prompt = '''
I have a remote sensing image about tennis court. Given the image attribute, please generate a description of the image based on the image attribute.
The image attribute I provide will begin with "Image Attribute:". The description you generate should begin with "Description:". 
The description should be concise and informative and contain all attributes. Avoid adding imaginary things. Use the examples below for reference.

Image Attribute: {"tennis": {"number": 3, "color": "green", "ground": "green"}, "withered trees": "one side"}
Description: three green tennis courts on a green ground with withered trees on one side 

Image Attribute: "tennis": {"number": 2, "color": "dark-purple", "ground": "purple", "shadow": true}, "buildings": {"number": 1, "color": "gray", "position": "between"}, "tree": "one side"}
Description: two dark-purple tennis courts on a purple ground with shadows, a gray building between the courts, and trees on one side
'''

tennis_convert_template = '''
Image Attribute: {}
'''

tennis_llama_prompt = '''
I have a remote sensing image about tennis court. Given the attribute of the image and a modification caption to edit the image, please generate the description of the edited image based on the image attribute and modification caption. 
The image attribute I provide will begin with "Image Attribute:". The modified caption I provide will begin with “Modification caption:". The edited attribute you generate should begin with “Edited Description:". The description you generate should only talk about the edited image and not the original image. 
The description should be concise and informative and contain all attributes. Avoid adding imaginary things. Use the examples below for reference.

Image Attribute: {"tennis": {"number": 2, "color": "dark-cyan", "ground": "dark-cyan", "shadow": true}}
Modification caption: some lawn on one side of tennis court and change tennis courts colors from dark-cyan to dark-green and change tennis courts shadow from True to 0 and change tennis courts ground from dark-cyan to dark-green
Edited Description: two dark-green tennis courts on dark-green ground with some lawn on one side

Image Attribute:  {"tennis": {"number": 1, "color": "dark-green", "ground": "red", "shadow": true}, "tree": "around"}
Modification caption: some lawn on one side of tennis court and change tree from around to one side and change tennis courts numbers from 1 to 2 and change tennis courts ground from red to dark-green
Edited Description: two dark-green tennis courts on dark-green ground with some lawn on one side and trees on one side

Image Attribute: {"tennis": {"number": 1, "color": "dark-green", "ground": "beige"}, "tree": "one side"}
Modification caption: has no tree on one side of tennis court and change tennis courts colors from dark-green to dark-blue and change tennis courts numbers from 1 to 2 and change tennis courts ground from beige to dark-green
Edited Description: two dark-blue tennis courts on dark-green ground

Image Attribute: {"tennis": {"number": 1, "color": "green", "ground": "red"}, "buildings": {"number": 1, "color": "black", "position": "one side"}, "lawn": "one side", "tree": "around"}
Modification caption: change buildings colors from black to brown or black and change buildings numbers from 1 to 4 and change lawn from one side to two sides and some swimming pool on one side of tennis court
Edited Description: one green tennis court on red ground with four brown or black buildings on one side, lawn on two sides, and swimming pool on one side
'''

tennis_llama_template = '''
Image Attribute: {}
Modification caption: {}
'''

WHDLD_convert_prompt = '''
I have a remote sensing image. Given the image attribute, please generate a description of the image based on the image attribute.
The image attribute I provide will begin with "Image Attribute:". The description you generate should begin with “Description:". 
The description should be concise and informative and contain all attributes. Avoid adding imaginary things. Use the examples below for reference.

Image Attribute: {"topological": [["building", "vegetation", "surrounded by"], ["vegetation", "building", "surround"], ["building", "water", "blend"], ["water", "building", "blend"], ["vegetation", "water", "surround"], ["water", "vegetation", "surrounded by"]], "objects": ["building", "vegetation", "water"], "attributes": [["building", 2, "has"], ["vegetation", 1, "has"], ["water", 6, "has"]]}
Description: vegetation surround building, building blend water, vegetation surround water, two building, one vegetation, six water

Image Attribute: {"topological": [["building", "pavement", "blend"], ["pavement", "building", "blend"], ["building", "vegetation", "blend"], ["vegetation", "building", "blend"], ["building", "water", "semi-around by"], ["water", "building", "semi-around"], ["pavement", "vegetation", "blend"], ["vegetation", "pavement", "blend"], ["pavement", "water", "semi-surrounded by"], ["water", "pavement", "semi-surround"], ["vegetation", "water", "blend"], ["water", "vegetation", "blend"]], "objects": ["building", "pavement", "vegetation", "water"], "attributes": [["building", 1, "has"], ["pavement", 3, "has"], ["vegetation", 2, "has"], ["water", 1, "has"]]}
Description: building blend pavement, building blend vegetation, water semi-around building, pavement blend vegetation, water semi-around pavement, vegetation blend water, one building, three pavement, two vegetation, one water

Image Attribute: {"topological": [["building", "road", "near"], ["road", "building", "near"], ["building", "pavement", "blend"], ["pavement", "building", "blend"], ["building", "vegetation", "blend"], ["vegetation", "building", "blend"], ["building", "baresoil", "blend"], ["baresoil", "building", "blend"], ["building", "water", "blend"], ["water", "building", "blend"], ["road", "pavement", "near"], ["pavement", "road", "near"], ["road", "vegetation", "semi-surrounded by"], ["vegetation", "road", "semi-surround"], ["road", "baresoil", "blend"], ["baresoil", "road", "blend"], ["road", "water", "near"], ["water", "road", "near"], ["pavement", "vegetation", "blend"], ["vegetation", "pavement", "blend"], ["pavement", "baresoil", "blend"], ["baresoil", "pavement", "blend"], ["pavement", "water", "blend"], ["water", "pavement", "blend"], ["vegetation", "baresoil", "blend"], ["baresoil", "vegetation", "blend"], ["vegetation", "water", "blend"], ["water", "vegetation", "blend"], ["baresoil", "water", "blend"], ["water", "baresoil", "blend"]], "objects": ["building", "road", "pavement", "vegetation", "baresoil", "water"], "attributes": [["building", 6, "has"], ["road", 1, "has"], ["pavement", 4, "has"], ["vegetation", 5, "has"], ["baresoil", 1, "has"], ["water", 2, "has"]]}
Description: building near road, building blend pavement, building blend vegetation, building blend baresoil, building blend water, road near pavement, vegetation semi-surround road, road blend baresoil, road near water, pavement blend vegetation, pavement blend baresoil, pavement blend water, vegetation blend baresoil, vegetation blend water, baresoil blend water, six building, one road, four pavement, five vegetation, one baresoil, two water
'''

WHDLD_convert_template = '''
Image Attribute: {}
'''

WHDLD_llama_prompt = '''
I have a remote sensing image. Given the attribute of the image and a modification caption to edit the image, please generate the description of the edited image based on the image attribute and modification caption. 
The image attribute I provide will begin with "Image Attribute:". The modification caption I provide will begin with "Modification caption:". The edited attribute you generate should begin with “Edited Description:". The description you generate should only talk about the edited image and not the original image. 
The description should be concise and informative and contain all attributes. Avoid adding imaginary things. Use the examples below for reference.

Image Attribute: {"topological": [["building", "vegetation", "surrounded by"], ["vegetation", "building", "surround"], ["building", "water", "blend"], ["water", "building", "blend"], ["vegetation", "water", "surround"], ["water", "vegetation", "surrounded by"]], "objects": ["building", "vegetation", "water"], "attributes": [["building", 2, "has"], ["vegetation", 1, "has"], ["water", 6, "has"]]}
Modification caption: change the number of water from 6 to 3 and change the number of building from 2 to 1 and add some baresoil blend building as well as blend vegetation as well as blend water
Edited Description: vegetation surround vbuilding, building blend water, vegetation surround water, vegetation blend baresoil, building blend baresoil, water blend baresoil, one building, one vegetation, three water, some baresoil

Image Attribute: {"topological": [["building", "pavement", "blend"], ["pavement", "building", "blend"], ["building", "vegetation", "blend"], ["vegetation", "building", "blend"], ["building", "water", "semi-around by"], ["water", "building", "semi-around"], ["pavement", "vegetation", "blend"], ["vegetation", "pavement", "blend"], ["pavement", "water", "semi-surrounded by"], ["water", "pavement", "semi-surround"], ["vegetation", "water", "blend"], ["water", "vegetation", "blend"]], "objects": ["building", "pavement", "vegetation", "water"], "attributes": [["building", 1, "has"], ["pavement", 3, "has"], ["vegetation", 2, "has"], ["water", 1, "has"]]}
Modification caption: change the number of water from 1 to 4 and want more blocks vegetation and want more pavement and remove all building
Edited Description: pavement blend vegetation, water semi-around pavement, vegetation blend water, more than three pavement, more than two vegetation, four water

Image Attribute: {"topological": [["building", "road", "near"], ["road", "building", "near"], ["building", "pavement", "blend"], ["pavement", "building", "blend"], ["building", "vegetation", "blend"], ["vegetation", "building", "blend"], ["building", "baresoil", "blend"], ["baresoil", "building", "blend"], ["building", "water", "blend"], ["water", "building", "blend"], ["road", "pavement", "near"], ["pavement", "road", "near"], ["road", "vegetation", "semi-surrounded by"], ["vegetation", "road", "semi-surround"], ["road", "baresoil", "blend"], ["baresoil", "road", "blend"], ["road", "water", "near"], ["water", "road", "near"], ["pavement", "vegetation", "blend"], ["vegetation", "pavement", "blend"], ["pavement", "baresoil", "blend"], ["baresoil", "pavement", "blend"], ["pavement", "water", "blend"], ["water", "pavement", "blend"], ["vegetation", "baresoil", "blend"], ["baresoil", "vegetation", "blend"], ["vegetation", "water", "blend"], ["water", "vegetation", "blend"], ["baresoil", "water", "blend"], ["water", "baresoil", "blend"]], "objects": ["building", "road", "pavement", "vegetation", "baresoil", "water"], "attributes": [["building", 6, "has"], ["road", 1, "has"], ["pavement", 4, "has"], ["vegetation", 5, "has"], ["baresoil", 1, "has"], ["water", 2, "has"]]}
Modification caption: want less water and change the number of vegetation from 5 to 1 and change the number of building from 6 to 2 and remove all baresoil and pavement
Edited Description: building near road, building blend vegetation, building blend water, vegetation semi-surround road, road near water, vegetation blend water, two building, one road, one vegetation, less than two water
'''

WHDLD_llama_template = '''
Image Attribute: {}
Modification caption: {}
'''