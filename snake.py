# Используем библиотеки PyTorch и NumPy (устанавливаем их заранее)
# matplotlib.pyplot используется для отображения графического интерфейса
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt
# Игру мы представим в виде матрицы целых чисел.
# Каждая пустая ячейка в игре будет иметь значение 0, хвост змеи будет 1, и по мере приближения хвоста к голове значение клеток будет увеличиваться на ещё одну единицу. 
# Клетка с едой имеет значение -1. Таким образом, для змеи размера N клеток хвост будет равен 1, а голова — N
def do(snake: t.Tensor, action: int):

# topk выполняет расчёты только по одному измерению, поэтому мы "разглаживаем" тензор с помощью flatten и используем unravel_index, чтобы превратить в двумерное состояние
    positions = snake.flatten().topk(2)[1]
# pos_cur и pos_prew операция вернёт вектор, указывающий на текущее направление движения змеи
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in positions]
# Поворот на 90 градусов (0 - налево, 1 - прямо, 2 - направо)
    rotation = T([[0, -1], [1, 0]]).matrix_power(3 + action)
    pos_next = (pos_cur + (pos_cur - pos_prev) @ rotation) % T(snake.shape)

# Конец игры. Если значение клетки больше 0, значит змейка расположена в этой клетке
    if (snake[tuple(pos_next)] > 0).any():
# Начальная длина змейки 2. Вычитаем это число и получаем счёт
        return (snake[tuple(pos_cur)] - 2).item() 

# Еда. Сравниваем следующую позицию змейки с -1. Если они равны, ищем место со значением 0, чтобы расположить следующую еду.
    if snake[tuple(pos_next)] == -1:
# Функция multinominal(n) выбирает n случайных индексов из тензора с вероятностью, основанной на значении элемента.
# Используем .flatten и .to(t.float), чтобы можно было разместить еду в любой точке, где не находится змейка.
        pos_food = (snake == 0).flatten().to(t.float).multinomial(1)[0]
# Возвращаем к двумерному состоянию и обновляем значение клетки до -1
        snake[unravel(pos_food, snake.shape)] = -1
# Процедура уменьшения и увеличения длины змейки (передвижение). Однако, если змейка находится в ячейке >0, то уменьшения не происходит (змейка увеличивается на 1 ячейку, "кушает")
    else:
        snake[snake > 0] -= 1  
    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1

    


# Создание графического интерфейса
snake = t.zeros((32, 32), dtype=t.int)
snake[0, :3] = T([1, 2, -1]) 

fig, ax = plt.subplots(1, 1)
img = ax.imshow(snake)
action = {'val': 1}

# Вводим кнопки управления (играть нужно на английской раскладке клавиатуры)
action_dict = {'a': 0, 'd': 2}

fig.canvas.mpl_connect('key_press_event',
                       lambda e: action.__setitem__('val', action_dict[e.key]))

score = None
while score is None: 
    img.set_data(snake)
    fig.canvas.draw_idle()
    plt.pause(0.1) 
    score = do(snake, action['val']) 
    action['val'] = 1 

print("Player    Score\n------" ,score, "------")

