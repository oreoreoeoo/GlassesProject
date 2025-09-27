import cv2
import numpy as np
import os

# ========== 配置 ==========
img_dir = "images"
frame_shapes = {
    "1": os.path.join(img_dir, "new_frame1.png"),
    "2": os.path.join(img_dir, "new_frame2.png")
}
textures = {
    "1": os.path.join(img_dir, "texture1.jpg"),
    "2": os.path.join(img_dir, "texture2.jpg")
}
output_path = os.path.join(img_dir, "new_glasses.png")
OUTPUT_WIDTH = 900  # 输出宽度
# ==========================

def resize_image(img, width=OUTPUT_WIDTH):
    h, w = img.shape[:2]
    scale = width / w
    new_size = (width, int(h * scale))
    return cv2.resize(img, new_size)

def load_frame(frame_path):
    """读取新镜框图片并缩放"""
    frame = cv2.imread(frame_path)
    frame = resize_image(frame, width=OUTPUT_WIDTH)
    return frame

def generate_mask(frame, threshold=15):
    """从镜框图生成掩码"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask

def colorize_frame(frame, mask, color):
    """给镜框上色"""
    colored_layer = np.zeros_like(frame)
    colored_layer[:] = color
    return cv2.bitwise_and(colored_layer, colored_layer, mask=mask)

def apply_texture(frame, mask, texture_path, texture_color):
    """给镜框添加花纹并着色"""
    texture = cv2.imread(texture_path)
    texture = resize_image(texture, width=frame.shape[1])

    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    _, tex_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    color_layer = np.zeros_like(texture)
    color_layer[:] = texture_color
    colored_texture = cv2.bitwise_and(color_layer, color_layer, mask=tex_mask)

    return cv2.bitwise_and(colored_texture, colored_texture, mask=mask)

def combine_layers(base_frame, mask, color_layer, texture_layer):
    """合并颜色和花纹到新镜框"""
    frame_final = cv2.add(color_layer, texture_layer)
    return cv2.bitwise_and(frame_final, frame_final, mask=mask) + cv2.bitwise_and(base_frame, base_frame, mask=cv2.bitwise_not(mask))

def main():
    # 1. 用户选择新镜框
    print("请选择镜框形状：")
    print("1 - new_frame1.png")
    print("2 - new_frame2.png")
    frame_choice = input("输入 1 或 2: ").strip()

    base_frame = load_frame(frame_shapes[frame_choice])
    mask = generate_mask(base_frame)

    # 2. 用户选择镜框颜色
    print("\n请输入镜框整体颜色 (B,G,R)，例如红色(0,0,255)：")
    try:
        frame_color = tuple(map(int, input("镜框颜色: ").strip("()").split(",")))
    except:
        print("输入有误，使用默认红色 (0,0,255)")
        frame_color = (0, 0, 255)

    color_layer = colorize_frame(base_frame, mask, frame_color)

    # 3. 用户选择花纹及颜色
    print("\n请选择花纹：")
    print("1 - texture1.jpg")
    print("2 - texture2.jpg")
    texture_choice = input("输入 1 或 2: ").strip()

    print("\n请输入花纹颜色 (B,G,R)，例如金色(0,215,255)：")
    try:
        texture_color = tuple(map(int, input("花纹颜色: ").strip("()").split(",")))
    except:
        print("输入有误，使用默认白色 (255,255,255)")
        texture_color = (255, 255, 255)

    texture_layer = apply_texture(base_frame, mask, textures[texture_choice], texture_color)

    # 4. 合并最终效果
    result = combine_layers(base_frame, mask, color_layer, texture_layer)

    # 5. 预览和保存
    cv2.imshow("Preview", result)
    print("\n按任意键关闭预览并保存图片...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(output_path, result)
    print(f"\n✅ 新眼镜已生成: {output_path}")

if __name__ == "__main__":
    main()
