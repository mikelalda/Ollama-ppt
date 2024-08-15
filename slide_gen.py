import os
import copy

from PIL import Image
import requests
from io import BytesIO

from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE_TYPE

class SlideGen:

    def __init__(self, output_folder="generated", template=None):
        self.prs_t = Presentation(template)
        self.template = template
        self.prs = Presentation()
        self.output_folder = output_folder
    
    def add_slide(self, slide_data):
        prs = self.prs
        bullet_slide_layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(bullet_slide_layout)
        shapes = slide.shapes

        # Title
        title_shape = shapes.title
        title_shape.text = slide_data.get("title_text", "")

        # Body
        if "text" in slide_data:
            body_shape = shapes.placeholders[1]
            tf = body_shape.text_frame
            for bullet in slide_data.get("text", []):
                p = tf.add_paragraph()
                p.text = bullet
                p.level = 0

                if "p1" in slide_data:
                    p = tf.add_paragraph()
                    p.text = slide_data.get("p1")
                    p.level = 1

        if "img_path" in slide_data:
            cur_left = 6
            for num_img, img_path in enumerate(slide_data.get("img_path", [])):
                top = Inches(2)
                left = Inches(cur_left)
                height = Inches(4)
                if img_path.startswith('http'):
                    response = requests.get(img_path)
                    img = Image.open(BytesIO(response.content))
                    name = 'images/' + str(num_img) + '.png'
                    img.save(name)
                    pic = slide.shapes.add_picture(name, left, top, height=height)
                else:
                    pic = slide.shapes.add_picture(img_path, left, top, height=height)
                cur_left += 1
    
    def update_presentation_content(self, slide_data, num_slide):
        # Iterate through all slides
        prs=self.prs
        image_num = 0
        base_slides = len(self.prs_t.slides) - 1
        select_slide = num_slide%base_slides + 1
        
        bullet_slide_layout = prs.slide_layouts[5]
        slide = prs.slides.add_slide(bullet_slide_layout)


        for shp in self.prs_t.slides[select_slide].shapes:
            el = shp.element
            newel = copy.deepcopy(el)
            slide.shapes._spTree.insert_element_before(newel, 'p:extLst')


        # Update shapes in the slide
        for shape in slide.shapes:
            if shape.name == "title_text":
                shape.text = slide_data.get("title_text", "")

            # Update text
            elif shape.name == 'text':
                shape.text = ''
                tf = shape.text_frame
                for bullet in slide_data.get("text", []):
                    p = tf.add_paragraph()
                    p.text = bullet
                    p.level = 0

                    if "p1" in slide_data:
                        p = tf.add_paragraph()
                        p.text = slide_data.get("p1")
                        p.level = 1
            
            # Update images
            elif shape.name == 'imagen':
                if "img_path" in slide_data:
                    new_image_path = slide_data["img_path"][image_num]
                    left, top, width, height = shape.left, shape.top, shape.width, shape.height
                    
                    # Remove the old picture
                    slide.shapes._spTree.remove(shape._element)
                    
                    # Add the new picture
                    if new_image_path.startswith('http'):
                        response = requests.get(new_image_path)
                        img = Image.open(BytesIO(response.content))
                        temp_path = 'temp_image.png'
                        img.save(temp_path)
                        slide.shapes.add_picture(temp_path, left, top, width, height)
                        os.remove(temp_path)
                    else:
                        slide.shapes.add_picture(new_image_path, left, top, width, height)
                    image_num += 1

    def add_title_slide(self, title_page_data):
        # title slide
        prs = self.prs
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        if "title_text" in title_page_data:
            title.text = title_page_data.get("title_text")
        if "subtitle_text" in title_page_data:
            subtitle.text = title_page_data.get("subtitle_text")
    
    def update_presentation_title(self, title_page_data):
        prs = self.prs
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        template_slide = self.prs_t.slides[0]
        for shp in template_slide.shapes:
            el = shp.element
            newel = copy.deepcopy(el)
            slide.shapes._spTree.insert_element_before(newel, 'p:extLst')

         # Update shapes in the slide
        for shape in slide.shapes:
            if shape.name == "title_text":
                shape.text = title_page_data.get("title_text")
            if shape.name == "subtitle_text":
                shape.text = title_page_data.get("subtitle_text")

    def create_presentation(self, title_slide_info, slide_pages_data=[]):
        file_name = title_slide_info.get("title_text").\
            lower().replace(",", "").replace(" ", "-")
        file_name += ".pptx"
        file_name = os.path.join(self.output_folder, file_name)
        if self.template == None:
            self.add_title_slide(title_slide_info)
        else:
            self.prs.slide_width = self.prs_t.slide_width
            self.prs.slide_height = self.prs_t.slide_height
            self.update_presentation_title(title_slide_info)
        
        for num_slide, slide_data in enumerate(slide_pages_data):
            if self.template == None:
                self.add_slide(slide_data)
            else:
                self.update_presentation_content(slide_data, num_slide)


        self.prs.save(file_name)
        return file_name