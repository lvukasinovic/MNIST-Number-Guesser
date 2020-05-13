import pygame
from PIL import Image
from tensorflow import keras
import numpy as np

pygame.init()
model = keras.models.load_model("myModel.h5")  # Loads model created from GetModel.py

# Sets screen and some definitions
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Number Guesser")
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
font = pygame.font.SysFont('arial', 32)


# Saves image as a 28x28 png
def saveImage():
    pygame.image.save(screen, "num.png")
    image = Image.open('num.png')
    new_image = image.resize((28, 28))
    new_image.save('num.png')


# Uses model to guess number drawn
def guessNumber(image):
    img = Image.open(image).convert("L")
    img = np.resize(img, (28, 28, 1))
    imgArr = np.array(img)
    imgArr = imgArr.reshape(1, 28, 28, 1)
    number = model.predict_classes(imgArr)
    return number


def instructions():
    screen.fill(BLACK)  # Need black background to recognize numbers
    text = font.render("Press the space bar to guess the number", False, WHITE)
    screen.blit(text, (0, 0))
    text = font.render("Press backspace to clear the screen", False, WHITE)
    screen.blit(text, (0, 50))
    text = font.render("Hold left mouse to draw", False, WHITE)
    screen.blit(text, (0, 100))
    text = font.render("Press backspace to continue", False, WHITE)
    screen.blit(text, (0, 150))


# Main method
def main():
    instructions()
    isRunning = True
    while isRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # If user presses space, program will guess number
                    saveImage()
                    guess = guessNumber('num.png')
                    text = font.render("Number: " + str(guess[0]), False, (255, 255, 255))
                    screen.blit(text, (0, 0))
                if event.key == pygame.K_BACKSPACE:  # If user presses backspace, screen will erase
                    screen.fill((0, 0, 0))
            mouseX, mouseY = pygame.mouse.get_pos()  # Gets current mouse position
            if pygame.mouse.get_pressed() == (1, 0, 0):  # If left mouse down, you will draw on screen
                pygame.draw.rect(screen, WHITE, (mouseX, mouseY, 25, 25))
        pygame.display.update()


main()
