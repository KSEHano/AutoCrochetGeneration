from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import numpy as np

thread_weight = {2: "lace or thread",3:"super fine", 3.5: "fine or sport", 5:"Light", 7:"Bulky", 8: "Bulky", 10: "super chunky, jumbo or super bulky"}
us_hooks = {2: "",3:"", 3.5: "(US: E)",5 : "(US: H)",7 : "",8 :"(US: L)", 10:"(US: N/P)"}

def print_pattern(instructions: dict, sample_points: dict, prev_rows: int = 0):
    repetition_start = 0
    counter = 0
    text = []

    
    
    for ind, row in instructions.items():
        if ind == 0:
            text.append(f'Row {ind + 1}: {len(sample_points[ind + 1])}sc in mR [{len(sample_points[ind + 1])}]')
        else:
            try:
                nextrow = instructions[ind+1]
            except KeyError:
                nextrow = None     
            

            if row == nextrow:
                if counter == 0:
                    repetition_start = ind
                counter += 1
                

            else:
                if counter == 0:
                    # drawString line with number
                    rowt = f'Row {ind + 1 + prev_rows}: {row} [{len(sample_points[ind + 1])}]'
                    pass
                
                else:
                    # calculate lines
                    rowt = f'Row {repetition_start + 1 + prev_rows} - {ind + 1 + prev_rows}: {row} [{len(sample_points[ind + 1])}]'

                    pass
                text.append(f'{rowt}')
                counter = 0

    return text


def print_joining_pattern(instructions: dict, sample_points: dict, prevStichNum, prev_rows):
    text = []
    #start where pattern is closest to prev stitchnumber

    r_len = np.array([len(x) for ind, x in sample_points.items()])
    idx = (np.abs(r_len - prevStichNum)).argmin()
    new_instructions  ={}
    new_sample_points = {}

    text.append('Contiue frome the legs by crocheting the legs together, you might have to sew them together if there are loose stitches between the legs.')
    

    for key, value in instructions.items():
        if key > idx:
            new_instructions[key-idx] = value

    for key, value in sample_points.items():
        if key > idx:
            new_sample_points[key-idx] = value


    text += print_pattern(new_instructions, new_sample_points, prev_rows)

    return text

def create_pattern_preamble(name,path,  imagePath, stitchwidth):
    #preamble
    key = ["sc - single crochet", "incX - increase (crochet X stitches in  one)", "decX - (invisible) decrease (crochet X stitches together)", "mR - magical ring", "() - repeat instructions in parentheses"]
    # open file
    doc = SimpleDocTemplate(path ,pagesize=A4,
                        rightMargin=72,leftMargin=72,
                        topMargin=72,bottomMargin=18)

    
    im = Image(imagePath, 7*cm, 7*cm)
    
    Story = []
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, leading = 60))
    
    Story.append(Paragraph(name, styles["Heading1"]))
    Story.append(Spacer(1, 12))
    Story.append(im)



    for i in key:
        Story.append(Paragraph(i, styles["Normal"])) #is it in different lines?

    Story.append(Spacer(1, 12))
    
    text = f'Use a {stitchwidth}mm {us_hooks[stitchwidth]} hook and fitting yarn ({thread_weight[stitchwidth]})'
    Story.append(Paragraph(text, styles["Normal"]))

    Story.append(Spacer(1, 12))
    text = 'Instructions:'
    Story.append(Paragraph(text, styles["Normal"]))

    return doc, Story, styles


def create_pattern_file(name, path, imagePath, instructions:dict, sample_points: dict, stitchwidth):
    #following https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/
    
    doc, Story, styles = create_pattern_preamble(name,path,  imagePath, stitchwidth)


    text = print_pattern(instructions, sample_points)
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    Story.append(Spacer(1, 12))

    text = 'Fasten off, stuff and sew together.'
    Story.append(Paragraph(text, styles["Normal"]))

    Story.append(Spacer(1, 12))

    text = "As this is a part of a Master's thesis please consider answering some questions about your experience"
    Story.append(Paragraph(text, styles["Normal"]))
    
    text = 'Anwer this <a href="https://forms.gle/9d6sXkCzC22Du4M59" color="blue"> questionnaire with picture</a> if you want to upload a picture (You need to log in with a Google account)'
    Story.append(Paragraph(text, styles["Normal"]))

    text = 'or this  <a href="https://forms.gle/7jpenij9XuzhwyvT8" color="blue"> questionnaire NO picture</a> when you want do not want to log in with a Google account.'
    Story.append(Paragraph(text, styles["Normal"]))
    doc.build(Story)



def create_rabbit_pattern_file(names:list, path, imagePath, instructions:list, sample_points: list, stitchwidth):

    doc, Story, styles = create_pattern_preamble(names[0],path,  imagePath, stitchwidth)

    Story.append(Spacer(1, 12))
    
    #ears
    text = f'{names[1]} (2x)'
    Story.append(Paragraph(text, styles["Normal"]))

    text = print_pattern(instructions[0], sample_points[0])
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    text = 'Fasten off and sew together.'
    Story.append(Paragraph(text, styles["Normal"]))

    Story.append(Spacer(1, 12))

    #head
    text = f'{names[2]}'
    Story.append(Paragraph(text, styles["Normal"]))

    text = print_pattern(instructions[1], sample_points[1])
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    text = 'Fasten off, stuff and sew together.'
    Story.append(Paragraph(text, styles["Normal"]))

    text = 'Sew ears to the head and add a face on the head.'
    Story.append(Paragraph(text, styles["Normal"]))

    Story.append(Spacer(1, 12))

    #arms
    text = f'{names[3]} (2x)' #added after first make pointed it out
    Story.append(Paragraph(text, styles["Normal"]))

    text = print_pattern(instructions[2], sample_points[2])
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    text = 'Fasten off, stuff and sew together.'
    Story.append(Paragraph(text, styles["Normal"]))
    
    Story.append(Spacer(1, 12))

    #tail
    text = f'{names[4]}'
    Story.append(Paragraph(text, styles["Normal"]))

    text = print_pattern(instructions[3], sample_points[3])
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    text = 'fasten off, stuff and sew together'
    Story.append(Paragraph(text, styles["Normal"]))
    
    Story.append(Spacer(1, 12))

    #legs
    text = f'{names[5]} (2x)'
    Story.append(Paragraph(text, styles["Normal"]))

    text = print_pattern(instructions[4], sample_points[4])
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    text = 'Fasten off the first leg but not the second and continue with the body.'
    Story.append(Paragraph(text, styles["Normal"]))

    Story.append(Spacer(1, 12))
    #body

    text = f'{names[6]}'
    Story.append(Paragraph(text, styles["Normal"]))

    text = print_joining_pattern(instructions[5], sample_points[5], 2* len(list(sample_points[4].values())[-1]), len(instructions[4])-1)
    for row in text:
        Story.append(Paragraph(row, styles["Normal"]))

    text = 'Fasten off, stuff and sew together.'
    Story.append(Paragraph(text, styles["Normal"]))

    Story.append(Spacer(1, 12))



    text = 'sew arms, head and tail to the body.'

    Story.append(Paragraph(text, styles["Normal"]))


    Story.append(Spacer(1, 12))


    text = 'Anwer this <a href="https://forms.gle/9d6sXkCzC22Du4M59" color="blue"> questionnaire with picture</a> if you want to upload a picture (You need to log in with a Google account)'
    Story.append(Paragraph(text, styles["Normal"]))

    text = 'or this  <a href="https://forms.gle/7jpenij9XuzhwyvT8" color="blue"> questionnaire NO picture</a> when you want do not want to log in with a Google account.'
    Story.append(Paragraph(text, styles["Normal"]))
    doc.build(Story)

    