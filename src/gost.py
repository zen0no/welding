

class GOSTRegistry:
    def __init__(self):
        self.gosts = {'fn': lambda *args, **kwargs: True, 'child': dict()}

    def _find_gost(self, id):
        verts = [self.gosts]
        while verts:
            v = verts.pop()
            if id in v['child'].keys():
                return v['child'][id]
            else:
                verts.extend(v['child'].items())
        return None

    def register(self, id, parent_id=None):
        def wrapper(func):
            if parent_id is not None:
                gost = self._find_gost(parent_id)
                if gost is None:
                    raise KeyError('Gost not found')
            else:
                gost = self.gosts
            
            gost['child'][id] = {'fn': func,
                                'childs': dict()}                
        return wrapper




register = GOSTRegistry()


