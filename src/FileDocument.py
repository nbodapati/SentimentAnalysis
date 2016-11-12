import os


class FileDocument:
    '''
    This is an implementation of the abstract document class to represent a document in the file system.
    '''
    def __init__(self, filePath):
        self.filePath = filePath

    def getContent(self):
        with open(self.filePath, 'r') as doc:
            content = doc.read()
        return content
