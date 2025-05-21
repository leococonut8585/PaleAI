from PIL import Image, ImageDraw, ImageColor


def generate_monkey_icon(color1: str, color2: str, gender: str) -> Image.Image:
    """Generate a simple two-color monkey avatar.

    The design roughly follows the sample images under ``static/pic`` (Default.png, Answer.png, Thinking.png).
    Only ``color1`` and ``color2`` are used for all strokes and fills.
    Gender controls minor features like eyelashes for a feminine style.
    """
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    c1 = ImageColor.getrgb(color1) + (255,)
    c2 = ImageColor.getrgb(color2) + (255,)

    # head and ears
    draw.ellipse((48, 48, 208, 208), fill=c1)
    draw.ellipse((24, 96, 72, 144), fill=c1)
    draw.ellipse((184, 96, 232, 144), fill=c1)

    # eyes and gender accents
    draw.ellipse((96, 104, 112, 120), fill=c2)
    draw.ellipse((144, 104, 160, 120), fill=c2)
    if gender == "女性":
        # eyelashes for a feminine style
        draw.rectangle((94, 98, 114, 100), fill=c2)
        draw.rectangle((142, 98, 162, 100), fill=c2)
    elif gender == "男性":
        # thicker eyebrows for a masculine look
        draw.rectangle((90, 96, 118, 100), fill=c2)
        draw.rectangle((138, 96, 166, 100), fill=c2)
    else:
        # neutral small brows
        draw.rectangle((94, 100, 112, 102), fill=c2)
        draw.rectangle((148, 100, 166, 102), fill=c2)

    # nose and mouth
    draw.rectangle((124, 136, 132, 160), fill=c2)
    draw.rectangle((112, 160, 144, 168), fill=c2)
    draw.rectangle((96, 168, 160, 176), fill=c2)

    return img
