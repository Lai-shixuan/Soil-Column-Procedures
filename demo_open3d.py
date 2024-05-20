import API_functions.dying as dy
import numpy as np

temp_array = np.random.randint(0, 2, (100, 100, 100))
(tuplea, tupleb) = dy.dying_color(temp_array, 3, 5)