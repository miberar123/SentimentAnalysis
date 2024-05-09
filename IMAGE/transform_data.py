import os
import shutil

class2folder = {
    "0": "Anger",
    "1": "Contempt",
    "2": "Disgust",
    "3": "Fear",
    "4": "Happy",
    "5": "Neutral",
    "6": "Sad",
    "7": "Surprise",
}
source_images_dir = "data/valid/images"
source_labels_dir = "data/valid/labels"
target_images_dir = "data/test/images"
target_labels_dir = "data/test/labels"


def join_test_valid():
    """
    Moves from valid to test
    """

    for image_file in os.listdir(source_images_dir):
        try:
            target_image_path = os.path.join(target_images_dir, image_file)
            if not os.path.exists(target_image_path):
                shutil.move(
                    os.path.join(source_images_dir, image_file), target_images_dir
                )
            else:
                print(
                    f"[WARNING] {image_file} already exists in {target_images_dir}. Skipping move."
                )
                f"[INFO] Moved {image_file} from {source_images_dir} to {target_images_dir}"

        except Exception as e:
            print(
                f"[ERROR] Could not move {image_file} from {source_images_dir} to {target_images_dir}: {e}"
            )
    for label_file in os.listdir(source_labels_dir):
        target_label_path = os.path.join(target_labels_dir, label_file)
        if not os.path.exists(target_label_path):
            shutil.move(os.path.join(source_labels_dir, label_file), target_labels_dir)
            print(
                f"[INFO] Moved {label_file} from {source_labels_dir} to {target_labels_dir}"
            )
        else:
            print(
                f"[WARNING] {label_file} already exists in {target_labels_dir}. Skipping move."
            )

    valid_folder = "data/valid"
    try:
        shutil.rmtree(valid_folder)
        print(f"[INFO] Successfully removed the folder: {valid_folder}")
    except Exception as e:
        print(f"[ERROR] Could not remove the folder {valid_folder}: {e}")


def reorganise_data(source_dir: str, target_dir: str):
    """
    Label it nicely

    Args:
        source_dir (str): from
        target_dir (str): to

    Raises:
        ValueError: if path is wrong
        IndexError: if txt does not have a correct label
    """
    if not os.path.exists(target_dir):
        print(f"[INFO] Target directory {target_dir} does not exist.")
        raise ValueError("Target directory does not exist.")

    for class_id, folder_name in class2folder.items():
        target_folder = os.path.join(target_dir, folder_name)

        if not os.path.exists(target_folder):
            print(f"[INFO] Creating target folder: {target_folder}")
        os.makedirs(target_folder, exist_ok=True)

        label_files = [
            f
            for f in os.listdir(os.path.join(source_dir, "labels"))
            if f.endswith(".txt")
        ]

        for file_name in label_files:
            with open(os.path.join(source_dir, "labels", file_name), "r") as file:
                to_label = file.read(1)  # First char is label rest idk

                if str(class_id) == str(to_label):
                    base_name: str = file_name.split(".")[0]
                    image_file = base_name + (
                        ".png" if base_name.startswith("ff") else ".jpg"
                    )

                    shutil.move(
                        os.path.join(source_dir, "images", image_file), target_folder
                    )
                    print(f"[INFO] Moved {image_file} to {target_folder}")


if __name__ == "__main__":
    join_test_valid()

    reorganise_data("./data/test", "./data/test")
    reorganise_data("./data/train", "./data/train")
