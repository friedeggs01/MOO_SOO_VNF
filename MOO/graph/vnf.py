class VNF():
    def __init__(self, id: int,
                 type: int,
                 install_cost: float,
                 resource_need = 0) -> None:
        self.id = id
        self.type = type
        self.install_cost = install_cost
        self.resource_need = resource_need