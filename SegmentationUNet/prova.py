import matplotlib.pyplot as plt
import nibabel as nib


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


epi_img = nib.load(
    r"C:\Users\loren\OneDrive - Politecnico di Milano\Desktop\Lorenzo\Università\NECSTCamp\Progetto\Segmentation\Data\volumes 0-49/volume-0.nii.gz")
epi_img_data = epi_img.get_fdata()
print(epi_img_data.shape)
# this outputs (512, 512, 75) since there are 75 images of 512x512 which
# compose the file

slice_0 = epi_img_data[:, :, 74]
slice_1 = epi_img_data[:, :, 54]
slice_2 = epi_img_data[:, :, 4]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()
