from fits_control import read_fits_file, edit_fits_data, show_image

input_file = 'generated/Comb_5/Comb/Comb_5.fits'
image = read_fits_file(input_file)
print(image[456,380])
