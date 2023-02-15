class Node:

    def __init__(self):
        self.gain = 0
        self.value = ""
        self.children = []
        self.isLeaf = False
        self.predict = ""
        self.numberofYes = 0
        self.numberofNo = 0
        self.parent = None

    def insertChild(self, value):
        value.parent = self
        self.children.append(value)

    def printTree(root, depth=0):
        print()
        print("\t" * depth, end="")
        print("{}  [YES]={} [NO]={}".format(root.value, root.numberofYes, root.numberofNo), end="")
        if root.isLeaf:
            print(" -> ", root.predict)
        else:
            for child in root.children:
                child.printTree(depth + 1)

    def findLeaves(root, leafList):
        if root.isLeaf:
            return
        allChild = False
        for child in root.children:
            if child.isLeaf:
                allChild = True
            else:
                allChild = False
                break
        if allChild:
            leafList.append(root)
        for child in root.children:
                child.findLeaves(leafList)

    def __lt__(self, other):
        return self.gain < other.gain

