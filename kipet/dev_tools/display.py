"""
Development Tools for display purposes

For debugging and other purposes
"""

class Print():
    
    """A print class for debugging - it makes it simpler to turn debugging
    statements on and off during development and later bug finding
    """
    
    def __init__(self, verbose=False):
        
        self.verbose=verbose
        
    def __call__(self, string):
        
        if self.verbose:
            print(string)
        
        return None

if __name__ == '__main__':
    
    pass