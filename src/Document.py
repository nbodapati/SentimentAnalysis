class Document:
    '''
    This is an abstract document.
    '''
    def getContent(self):
        raise NotImplementedError("Abstract Document needs to be implemented")