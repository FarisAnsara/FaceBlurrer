from BlurFace import BlurFace
import os

def main():
    # Define model path
    model_path = os.path.join('detect_faces', 'training', 'weights', 'best.pt')

    # Define input image path
    input_image_path = os.path.join('FacesDataSet', 'faces', 'split', 'images', 'test', '0a0bdec4b07ca3c1.jpg')
    
    output_image_folder = os.path.join('FacesDataSet', 'faces', 'blurred_faces')
    os.makedirs(output_image_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(input_image_path))[0]
    blurred_filename = f"{filename}_blurred.jpg"
    output_image_path = os.path.join(output_image_folder, blurred_filename)
    blur_face = BlurFace(model_path, conf=0.325, device=0)
    
    try:
        face_detected = blur_face.blur_faces(input_image_path, output_image_path, blur_strength=(51, 51))
        if not face_detected:
            print("No faces were detected or blurred.")
        else:
            print(f"Faces successfully blurred and saved to {output_image_path}.")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
