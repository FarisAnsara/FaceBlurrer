import os
import argparse
from BlurFace import BlurFace

def main():
    parser = argparse.ArgumentParser(description="Blur faces in an image using a pre-trained model.")
    parser.add_argument(
        "--input_image_path", 
        type=str, 
        default=os.path.join('FacesDataSet', 'test_image', '0a572a8dedf1ad07.jpg'),
        help="Path to the input image. Default is a sample image."
    )
    parser.add_argument(
        "--output_image_path", 
        type=str, 
        default=None,
        help="Path to save the blurred image. If not provided, a default location will be used."
    )

    args = parser.parse_args()
    model_path = os.path.join('detect_faces', 'training', 'weights', 'best.pt')
    input_image_path = args.input_image_path

    if args.output_image_path:
        output_image_path = args.output_image_path
        valid_extensions = ('.jpg', '.jpeg', '.png')
        if not output_image_path.lower().endswith(valid_extensions):
            input_extension = os.path.splitext(input_image_path)[1]
            output_image_path = os.path.join(
                output_image_path,
                f"{os.path.splitext(os.path.basename(input_image_path))[0]}_blurred{input_extension}"
            )
    else:
        output_image_folder = os.path.join('FacesDataSet', 'images', 'blurred_faces')
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
