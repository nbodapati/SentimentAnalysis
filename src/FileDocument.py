import os


class FileDocument:
    '''
    This is an implementation of the abstract document class to represent a document in the file system.
    '''
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def getContent(self):
        with open(os.path.join(self.path, self.filename), 'r') as doc:
            content = doc.read()
        return content
