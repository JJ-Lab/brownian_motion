import cv2
import time

# Inicializar el video capture
cap = cv2.VideoCapture(0)

# Establecer las dimensiones del frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el detector de movimiento
detector = cv2.createBackgroundSubtractorMOG2()

# Inicializar la posición de la partícula
posicion_anterior = None

# Establecer la cantidad de tiempo que se desea capturar y la frecuencia para tomar medidas
tiempo_total = 1800  # 30 minutos en segundos
frecuencia = 180     # 3 minutos en segundos

# Inicializar el tiempo actual
tiempo_actual = time.time()

while(time.time() - tiempo_actual < tiempo_total):
    # Obtener el siguiente frame
    ret, frame = cap.read()

    # Aplicar el detector de movimiento
    fgmask = detector.apply(frame)

    # Obtener el contorno de la región de movimiento
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Buscar el contorno más grande
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Si se encontró un contorno, actualizar la posición de la partícula
    if max_contour is not None:
        # Obtener el centroide del contorno
        M = cv2.moments(max_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Si es la primera medida, establecer la posición anterior
        if posicion_anterior is None:
            posicion_anterior = (cx, cy)

        # Si han pasado 3 minutos, imprimir la posición actual
        if time.time() - tiempo_actual >= frecuencia:
            print("Posición actual: ({}, {})".format(cx, cy))
            tiempo_actual = time.time()

        # Actualizar la posición anterior
        posicion_anterior = (cx, cy)

    # Mostrar el frame
    cv2.imshow('frame', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

##Páginas útiles para opencv:
#https://docs.opencv.org/master/d9/df8/tutorial_root.html
#https://learnopencv.com/
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
#https://www.pyimagesearch.com/
#https://www.udemy.com/course/computer-vision-for-faces/

