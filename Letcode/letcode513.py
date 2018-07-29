"""
给定一个二叉树，在树的最后一行找到最左边的值
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 采用中序遍历的思想，记录最深层次的第一个值
def left_mid_right_search(root, prehigh, maxhigh, leftval):
    if not root:
        return
    left_mid_right_search(root.left, prehigh + 1, maxhigh, leftval)
    if prehigh > maxhigh[0]:
        maxhigh[0] = prehigh
        leftval[0] = root.val
    left_mid_right_search(root.right, prehigh + 1, maxhigh, leftval)


class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        prehigh = 0
        maxhigh = [-1]
        leftval = [0]

        left_mid_right_search(root, prehigh, maxhigh, leftval)

        return leftval[0]
