"""
The following is a simple algorithm template that matches the configuration for your algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy
import torch
import monai

from scipy.special import expit
from skimage import transform

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # Read the input
    input_oct_image = load_image_file_as_array(
        location=INPUT_PATH / "images/oct",
    )
    input_age_in_months = load_json_file(
        location=INPUT_PATH / "age-in-months.json",
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()
    # use GPU if available, otherwise use the CPU
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())

    # TODO: add your custom inference here
    # define the U-Net with monai
    self.model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(self.device)
    print(f'==> Model created')

    res = self.model.eval()  # set to inference mode
    print(f'==> Model evaluated')

    self.model.load_state_dict(
        torch.load(
            "./best_metric_model_segmentation2d_dict.pth",
            map_location=self.device,
        )
    )

    print("==> Weights loaded")
   
    image = SimpleITK.GetArrayFromImage(input_image)
    image = np.array(image)
    shape = image.shape

    # Pre-process the image
    image = transform.resize(image, (512, 512), order=3)  # resize all images to 512 x 512 in shape
    image = image.astype(np.float32) / 255.  # normalize
    image = image.transpose((2, 0, 1))  # flip the axes and bring color to the first channel
    image = torch.from_numpy(image).to(self.device).reshape(1, 3, 512, 512)

    # Do the forward pass
    out = self.model(image).squeeze().data.cpu().numpy()

    # Post-process the image
    out = transform.resize(out, shape[:-1], order=3)
    out = (expit(out) > 0.99)  # apply the sigmoid filter and binarize the predictions
    out = (out * 255).astype(np.uint8)
    out = SimpleITK.GetImageFromArray(out)  # convert numpy array to SimpleITK image for grand-challenge.org    
    
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/vessel-segmentation",
        array=out,
    )
    print("==> Prediction done")
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
