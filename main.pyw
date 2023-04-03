from tkinter import Button, Canvas, Entry, Frame, Label, Radiobutton, StringVar, Tk
from tkinter.messagebox import showerror
from PIL import Image, ImageTk
from neural_network import *
from math import sqrt


WIDTH = 500
HEIGHT = 500
FONT = "Consolas 12"
MARKS_DIST = 10
MARKS_RADIUS = 5
COLORS = ["red", "green", "blue"]


class Window:
    def __init__(self):
        self.__window = Tk()
        self.__window.title("Shapes")
        self.__window.resizable(width=False, height=False)
        self.__canvas = Canvas(
            self.__window,
            width=WIDTH,
            height=HEIGHT,
            bg="white",
        )
        self.__canvas.grid(
            row=0,
            column=0,
        )
        frame = Frame(
            self.__window,
            borderwidth=0,
        )
        frame.grid(
            row=0,
            column=1,
            sticky="nw",
        )
        self.__color = StringVar()
        for i in range(len(COLORS)):
            Radiobutton(
                frame,
                text=COLORS[i],
                font=FONT,
                variable=self.__color,
                value=COLORS[i],
            ).grid(
                row=i,
                column=0,
                sticky="nw",
            )
        self.__color.set(COLORS[0])
        Button(
            frame,
            text="Clear",
            font=FONT,
            command=self.__clear,
        ).grid(
            row=len(COLORS),
            column=0,
            sticky="nw",
        )
        frame2 = Frame(
            frame,
            borderwidth=0,
        )
        frame2.grid(
            row=len(COLORS)+1,
            column=0,
            sticky="we",
        )
        self.__entry_layers = Entry(
            frame2,
            font=FONT,
            width=10,
        )
        self.__entry_layers.grid(
            row=0,
            column=0,
        )
        self.__entry_layers.insert("end", "2 8 8 3")
        Button(
            frame2,
            text="Update",
            font=FONT,
            command=self.__update_layers,
        ).grid(
            row=0,
            column=1,
            sticky="we",
        )
        self.__entry_run = Entry(
            frame2,
            font=FONT,
            width=10,
        )
        self.__entry_run.grid(
            row=1,
            column=0,
        )
        self.__entry_run.insert("end", "100000")
        Button(
            frame2,
            text="Run",
            font=FONT,
            command=self.__run,
        ).grid(
            row=1,
            column=1,
            sticky="we",
        )
        self.__label_precision = Label(
            frame,
            text="Precision : ?",
            font=FONT,
        )
        self.__label_precision.grid(
            row=len(COLORS)+2,
            column=0,
            sticky="nw",
        )
        self.__canvas.bind("<B1-Motion>", self.__mouse_left)
        self.__last_mouse_pos = (0, 0)
        self.__neural_network = None
        self.__x = []
        self.__y = []
        self.__update_layers()
        self.__window.mainloop()

    def __mouse_left(self, event):
        if len(self.__x) == 0:
            self.__clear()
        if sqrt((self.__last_mouse_pos[0]-event.x)**2+(self.__last_mouse_pos[1]-event.y)**2) > MARKS_DIST:
            self.__last_mouse_pos = (event.x, event.y)
            self.__x.append([event.x/WIDTH, event.y/HEIGHT])
            self.__y.append(COLORS.index(self.__color.get()))
            self.__canvas.create_oval(
                event.x-MARKS_RADIUS+2,
                event.y-MARKS_RADIUS+2,
                event.x+MARKS_RADIUS+2,
                event.y+MARKS_RADIUS+2,
                fill=self.__color.get(),
                width=0,
            )

    def __clear(self):
        self.__canvas.delete("all")
        del self.__x[:]
        del self.__y[:]

    def __update_layers(self):
        layers = []
        valid = 1
        for i in self.__entry_layers.get().split():
            try:
                layers.append(int(i))
            except:
                valid = 0
                showerror(
                    "Error",
                    "Invalid layer value",
                )
                break
        if valid:
            self.__neural_network = NeuralNetwork(layers)

    def __run(self):
        x = np.array(self.__x)
        y = np.zeros((len(self.__y), len(COLORS)))
        for i in range(y.shape[0]):
            y[i][self.__y[i]] = 1
        try:
            for i in range(int(self.__entry_run.get())):
                self.__neural_network.train(x, y, 1)
            self.__label_precision["text"] = f"Precision : {(1-self.__neural_network.cost)*100}%"
            inputs = np.array([[x/WIDTH, y/HEIGHT] for y in range(HEIGHT) for x in range(WIDTH)])
            self.__neural_network.forward(inputs)
            image = Image.new(mode="RGB", size=(WIDTH, HEIGHT))
            x, y = 0, 0
            for i in range(self.__neural_network.outputs.shape[0]):
                color = [0, 0, 0]
                color[self.__neural_network.outputs[i].argmax()] = 255
                image.putpixel((x, y), tuple(color))
                x += 1
                if x == WIDTH:
                    x = 0
                    y += 1                
            self.__clear()
            image_tk = ImageTk.PhotoImage(image)
            self.__canvas.image_tk = image_tk
            self.__canvas.create_image(2, 2, anchor="nw", image=image_tk)
        except:
            showerror(
                "Error",
                "Invalid train number",
            )


window = Window()
