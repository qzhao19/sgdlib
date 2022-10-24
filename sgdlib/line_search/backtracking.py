import numpy as np

class LineSearchBacktracking(object):
    def search(self, x, fx0, g, step, d, xp, gp, max_step, min_step, loss, line_search_params):

        # Decreasing and increasing factors
        decrease_factor = line_search_params.decrease_factor
        increase_factor = line_search_params.increase_factor

        if step <= 0:
            raise ValueError("'step' must be positive")
        
        
