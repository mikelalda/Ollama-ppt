import os

from PIL import Image
import requests
from io import BytesIO

from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE_TYPE

class SlideGen:

    def __init__(self, output_folder="generated", template=None):
        self.template = template
        self.prs = Presentation(template)
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
        base_slides = len(self.prs.slides) - 1
        slide = self.prs.slides[base_slides%num_slide + 1]
                
        # Update shapes in the slide
        for shape in slide.shapes:
            # Update text
            if shape.has_title:
                if shape.tilte in slide_data:
                    shape.title = slide_data.get("title_text", "")

            # Update text
            elif shape.has_text_frame:
                if shape.text in slide_data:
                    shape.text = ''
                    tf = shape.text
                    for bullet in slide_data.get("text", []):
                        p = tf.add_paragraph()
                        p.text = bullet
                        p.level = 0

                        if "p1" in slide_data:
                            p = tf.add_paragraph()
                            p.text = slide_data.get("p1")
                            p.level = 1
            
            # Update images
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                if f"image_{shape.shape_id}" in slide_data:
                    new_image_path = slide_data[f"image_{shape.shape_id}"]
                    left, top, width, height = shape.left, shape.top, shape.width, shape.height
                    slide.shapes._spTree.remove(shape._element)
                    slide.shapes.add_picture(new_image_path, left, top, width, height)

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

    def create_presentation(self, title_slide_info, slide_pages_data=[]):
        try:
            file_name = title_slide_info.get("title_text").\
                lower().replace(",", "").replace(" ", "-")
            file_name += ".pptx"
            file_name = os.path.join(self.output_folder, file_name)
            self.add_title_slide(title_slide_info)
            for num_slide, slide_data in enumerate(slide_pages_data):
                if self.template == None:
                    self.add_slide(slide_data)
                else:
                    self.update_presentation_content(slide_data, num_slide+1)


            self.prs.save(file_name)
            return file_name
        except Exception as e:
            raise e