import cv2
import numpy as np

img = cv2.imread('/Users/guirmiguel/Desktop/teste.jpg') #Caminho da imagem

#### Aqui é possível aumentar ou diminuir o brilho e o contraste da imagem, embora isso não seja necessário para
#### a maioria dos casos

#a = 1.0 #contraste
#b = 0.0 #brilho

#n_img = np.zeros(img.shape, img.dtype)

#for y in range(img.shape[0]):
 #   for x in range(img.shape[1]):
  #      for c in range(img.shape[2]):
   #         n_img[y,x,c] = np.clip(a*img[y,x,c] + b, 0, 255)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convertendo a coloração da imagem de BGR para HSV

red1 = cv2.inRange(hsv, (10, 50, 50), (15, 255, 255))
red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255)) #filtrando as cores diferentes de vermelho, laranja e amarelo
red = red1 | red2

cv2.imshow('imagem', img) #Imagem inicial
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow('imagem', n_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


cv2.imshow('filtros', red) #Resultado final
cv2.waitKey(0)
cv2.destroyAllWindows()
