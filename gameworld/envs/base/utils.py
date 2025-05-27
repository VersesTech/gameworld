from PIL import Image, ImageDraw


def parse_image(path, size=(40, 40), threshold=30):
    img = Image.open(path).convert("RGBA")
    img = img.resize(size, resample=Image.NEAREST)

    bg_color = img.getpixel((0, 0))

    def is_close_to_bg(pixel, bg):
        return all(abs(pixel[i] - bg[i]) < threshold for i in range(3))

    new_data = [
        (0, 0, 0, 0) if is_close_to_bg(pixel, bg_color) else pixel
        for pixel in img.getdata()
    ]
    img.putdata(new_data)
    return img


def make_ball(size=(10, 10), color=(0, 0, 0, 255)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, size[0] - 1, size[1] - 1], fill=color)
    return img
