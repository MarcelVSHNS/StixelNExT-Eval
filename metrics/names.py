import yaml

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Stixel:
    def __init__(self, x, y_t, y_b, depth=8.0):
        self.column = x
        self.top = y_t
        self.bottom = y_b
        self.depth = depth
        self.scale_by_grid()

    def __repr__(self):
        return f"{self.column},{self.top},{self.bottom},{self.depth}"

    def scale_by_grid(self, grid_step=config['grid_step']):
        self.column = self.column * grid_step
        self.top = self.top * grid_step
        self.bottom = self.bottom * grid_step