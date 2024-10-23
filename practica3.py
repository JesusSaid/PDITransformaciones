import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.autolayout"] = True

# IMAGE NEGATIVES
img = cv2.imread('bajoContraste.jpg')

'''''
def negative(r):
    s = 255 - r
    return s

img_neg = negative(img)

# Graficar la imagen original y la imagen negativa
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(122)
plt.imshow(img_neg)
plt.title('Imagen Negativa')
plt.axis('off')

plt.show()

# Graficar la transformación de intensidad
x_values = np.linspace(0, 255, 500)
y_values = negative(x_values)

plt.plot(x_values, y_values)
plt.xlabel('Input Intensity Levels')
plt.ylabel('Output Intensity Levels')
plt.grid(True)
plt.show()
'''''
# Función para escalar la imagen
def scale_image(input_img):
    input_img = input_img / np.max(input_img)
    input_img = (input_img * 255).astype('int')
    return input_img

# Función para graficar los resultados
def plot_results(input_img, output_img, x_values, y_values, save_as):
    plt.figure(figsize = (36,12))

    plt.subplot(131)
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(132)
    plt.plot(x_values, y_values)
    plt.xlabel('Input Pixels')
    plt.ylabel('Output Pixels')
    plt.grid(True)

    plt.subplot(133)
    plt.imshow(output_img)
    plt.title('Transformed Image')
    plt.axis('off')

    plt.savefig(save_as + '.png')
    plt.show()
'''''
# Llamada a la función para graficar los resultados
plot_results(img, img_neg, x_values, y_values, 'negative')
'''''
'''''
# LOG TRANSFORMATION
#img = cv2.imread('altoContraste.jpeg')
#plt.imshow(img)
def logTransform(r, c=1):
    s = c*np.log(1.0+r)
    return s

img_log = logTransform(img)
img_log_scaled = scale_image(img_log)

x_values = np.linspace(0,255,500)
y_values = logTransform(x_values)

plot_results(img, img_log_scaled, x_values, y_values, 'log')

#GAMMA TRANSFORMATIONS
#img = cv2.imread('altoContraste.jpeg')
plt.figure(figsize = (12,8))
#plt.imshow(img)
plt.axis('off')

def gammaTransform(r, gamma, c=1):
    s = c * (r**gamma)
    return s

#GAMMA < 1
img_gamma = gammaTransform(img, 0.4)
img_gamma_scaled = scale_image(img_gamma)

x_values = np.linspace(0,255,500)
y_values = gammaTransform(x_values, 0.4)

plot_results(img, img_gamma_scaled, x_values, y_values, 'gamma_0_4')

def performGammaTransform(input_img, gammaValue):
    img_gamma = gammaTransform(input_img, gammaValue)
    img_gamma_scaled = scale_image(img_gamma)

    x_values = np.linspace(0,255,500)
    y_values = gammaTransform(x_values, gammaValue)

    plot_results(input_img, img_gamma_scaled, x_values, y_values, "gamma" + str(gammaValue))

    return img_gamma_scaled

final_images = []

for gammaValue in [0.6, 0.4, 0.3]:
    final_images.append(performGammaTransform(img, gammaValue))

# COMPARING THE OUTPUTS
plt.figure(figsize = (24,12))

plt.subplot(221)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.imshow(img)
plt.title('Gamma = 0.6')
plt.axis('off')

plt.subplot(223)
plt.imshow(img)
plt.title('Gamma = 0.4')
plt.axis('off')

plt.subplot(224)
plt.imshow(img)
plt.title('Gamma = 0.3')
plt.axis('off')

plt.tight_layout()
plt.savefig('gamma.png')

# WITH GAMMA > 1
#img = cv2.imread('altoContraste.jpeg')
plt.figure(figsize = (10,6))
#plt.imshow(img)
plt.axis('off')

final_images_arial = []

for gammaValue in [3.0, 4.0, 5.0]:
    final_images_arial.append(performGammaTransform(img, gammaValue))

# COMPARING THE OUTPUTS
plt.figure(figsize = (24,12))

plt.subplot(221)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.imshow(final_images_arial[0])
plt.title('Gamma = 3.0')
plt.axis('off')

plt.subplot(223)
plt.imshow(final_images_arial[1])
plt.title('Gamma = 4.0')
plt.axis('off')

plt.subplot(224)
plt.imshow(final_images_arial[2])
plt.title('Gamma = 5.0')
plt.axis('off')

plt.tight_layout()
plt.savefig('gamma_arial.png')


#ESTIRAMIENTO DE CONTRASTE

#img = cv2.imread('altoContraste.jpeg')

plt.figure(figsize = (10,6))
#plt.imshow(img)
plt.axis('off')

def piecewiseLinear(r, r1, s1, r2, s2):
    if r < r1:
        s = (s1 / r1) * r
    elif r > r1 and r < r2:
        s = ((s2 - s1) / (r2 -r1)) * (r - r1) + s1
    else:
        s = ((255 - s2) / (255 - r2)) * (r - r2) + s2
    
    return int(s)

piecewiseLinearVec = np.vectorize(piecewiseLinear)
x_values = np.linspace(0,255,500)
y_values = piecewiseLinearVec(x_values,80,30,150,190)

plt.plot(x_values, y_values)
transformed_im = piecewiseLinearVec(img,80,30,150,190)
plot_results(img, transformed_im, x_values, y_values, 'piecewisel')

m = np.mean(img)

transformed_im2 = piecewiseLinearVec(img, m, 0, m, 255)

x_values = np.linspace(0,255,500)
y_values = piecewiseLinearVec(x_values, m, 0, m, 255)

plot_results(img, transformed_im2, x_values, y_values, 'piecewise2')

#REBANADAS DE NIVEL DE INTENCIDAD
#img = cv2.imread('altoContraste.jpeg')

plt.figure(figsize = (12,8))
#plt.imshow(img)
plt.axis('off')

def intensityLevelTransform(r):
    if r > 180 and r < 210:
        return 255
    else:
        return 0

intensityLevelTransformVec = np.vectorize(intensityLevelTransform)
transform1 = intensityLevelTransformVec(img)

x_values = np.linspace(0,255,500)
y_values = intensityLevelTransformVec(x_values)

plt.plot(x_values,y_values)
plot_results(img, transform1, x_values, y_values, 'intensity_level1')

def intensityLevelTransform2(r):
    if r > 180 and r < 210:
        return 240
    else:
        return r
    
intensityLevelTransform2vec = np.vectorize(intensityLevelTransform2)
transform2 = intensityLevelTransform2vec(img)

x_values = np.linspace(0,255,500)
y_values = intensityLevelTransform2vec(x_values)
plot_results(img, transform2, x_values, y_values, 'intensity_level2')
'''''
##REABANADA DE PLANO DE BIT

dollar_img = cv2.imread('pocaIluminacion.jpeg',0)
plt.imshow(dollar_img, cmap = "gray")
plt.axis('off')

def bitPlaneSlicing(r, bit_plane):
    dec = np.binary_repr(r, width = 8)
    return int(dec[8 - bit_plane])

bitPlaneSlicingVec = np.vectorize(bitPlaneSlicing)
eight_bitplace = bitPlaneSlicingVec(dollar_img, bit_plane = 8)
plt.imshow(eight_bitplace, cmap = "gray")

bit_planes_dict = {}
for bit_plane in np.arange(8,0,-1):
    bit_planes_dict['bit_plane_' + str(bit_plane)] = bitPlaneSlicingVec(dollar_img, bit_plane = bit_plane)

plt.figure(figsize = (24,12))

plt.subplot(331)
plt.imshow(dollar_img, cmap = "gray")
plt.title('Original Image')
plt.axis('off')

plt.subplot(332)
plt.imshow(bit_planes_dict['bit_plane_8'], cmap = "gray")
plt.title('bit_plane_8')
plt.axis('off')

plt.subplot(333)
plt.imshow(bit_planes_dict['bit_plane_7'], cmap = "gray")
plt.title('bit_plane_7')
plt.axis('off')

plt.subplot(334)
plt.imshow(bit_planes_dict['bit_plane_6'], cmap = "gray")
plt.title('bit_plane_6')
plt.axis('off')

plt.subplot(335)
plt.imshow(bit_planes_dict['bit_plane_5'], cmap = "gray")
plt.title('bit_plane_5')
plt.axis('off')

plt.subplot(336)
plt.imshow(bit_planes_dict['bit_plane_4'], cmap = "gray")
plt.title('bit_plane_4')
plt.axis('off')

plt.subplot(337)
plt.imshow(bit_planes_dict['bit_plane_3'], cmap = "gray")
plt.title('bit_plane_3')
plt.axis('off')

plt.subplot(338)
plt.imshow(bit_planes_dict['bit_plane_2'], cmap = "gray")
plt.title('bit_plane_2')
plt.axis('off')

plt.subplot(339)
plt.imshow(bit_planes_dict['bit_plane_1'], cmap = "gray")
plt.title('bit_plane_1')
plt.axis('off')

plt.tight_layout()
plt.savefig('bit_plane_slicing.png')
