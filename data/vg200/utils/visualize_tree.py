from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

class decisionnode:
    def __init__(self, value, gt_classes, class_names, isleaf=False, leafcounts=0, maxlevel=1):
        self.childs = []
        self.value = value
        self.name = class_names[gt_classes[self.value]] + '_' + str(self.value) if self.value >= 0 else 'scene'
        self.isleaf = isleaf
        self.leafcounts = leafcounts
        self.maxlevel = maxlevel

    def addchild(self, child):
        self.childs.append(child)


class RelationTree:
    def __init__(self, gt_classes, class_names, basewidth=150, basedepth=100):
        self.basewidth = basewidth
        self.basedepth = basedepth
        self.gt_classes = gt_classes
        self.class_names = class_names
        self.root = None

    def gentree(self, tree_data, rootvalue=-1):
        self.root = decisionnode(rootvalue, self.gt_classes, self.class_names)

        def swap_gentree(node):
            if node.value in tree_data:
                results = tree_data[node.value]

            else:  ## is leaf
                return decisionnode(node.value, self.gt_classes, self.class_names, isleaf=True, maxlevel=1)

            # 程序运行到这里，说明是非叶子节点
            # 对非叶子节点进行其下包含的叶子节点进行统计（leafcounts）
            # 该节点之下最深的深度maxlevel收集
            maxlevel = 1
            for each in results:
                entrynode = swap_gentree(decisionnode(each, self.gt_classes, self.class_names))
                if (entrynode.isleaf):
                    node.leafcounts += 1
                else:
                    node.leafcounts += entrynode.leafcounts

                if (entrynode.maxlevel > maxlevel):
                    maxlevel = entrynode.maxlevel
                node.addchild(entrynode)

            node.maxlevel = maxlevel + 1
            return node

        swap_gentree(self.root)

    def draweachnode(self, tree, ax, x, y):
        ax.text(x, y,
                tree.name,
                bbox=dict(facecolor='yellow', alpha=0.5),
                fontsize=14, color='black')
        # draw.text((x, y), tree.name, (0, 0, 0))

        if not tree.childs:
            return

        childs_leafcounts = [child.leafcounts if child.leafcounts else 1 for child in tree.childs]

        leafpoint = x - sum(childs_leafcounts) * self.basewidth / 2

        cumpoint = 0
        for childtree, point in zip(tree.childs, childs_leafcounts):
            centerpoint = leafpoint + self.basewidth * cumpoint + self.basewidth * point / 2
            cumpoint += point
            # draw.line((x, y, centerpoint, y + self.basedepth), (255, 0, 0))
            ax.arrow(x, y, centerpoint - x, self.basedepth, color='black')
            self.draweachnode(childtree, ax, centerpoint, y + self.basedepth)

    def drawTree(self):
        width = self.root.leafcounts * self.basewidth + self.basewidth
        depth = self.root.maxlevel * self.basedepth + self.basedepth
        img = Image.new(mode="RGB", size=(width, depth), color=(255, 255, 255))
        fig, ax = plt.subplots(figsize=(50, 25))
        ax.imshow(img, aspect='equal')
        # draw = ImageDraw.Draw(img)
        self.draweachnode(self.root, ax, width / 2, 20)

        ax.imshow(img, aspect='equal')
        # img.save(filename)


def visual_image_with_trees(db, im_refs, dbidx, classes, predicates):
    imdb = db['imdb']
    this_im = imdb[dbidx]
    w, h = this_im['w'], this_im['h']
    image = im_refs[dbidx]
    image = image[:, :h, :w]
    image = image.transpose((1, 2, 0))
    # transform from BGR to RGB
    image = image[:, :, (2, 1, 0)].astype(np.uint8)
    trees = this_im['trees']
    boxes = this_im['boxes']

    gt_relations = this_im['gt_relations']
    gt_classes = this_im['gt_classes']

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, aspect='equal')

    for i, box in enumerate(boxes):
        ax.add_patch(
            plt.Rectangle((box[0], box[1]),
                          box[2] - box[0],
                          box[3] - box[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        label_str = classes[int(gt_classes[i])]
        ax.text(box[0], box[1] - 2,
                label_str,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    for i, rel in enumerate(gt_relations):
        sub_label = classes[int(gt_classes[rel[0]])]
        obj_label = classes[int(gt_classes[rel[1]])]
        pred = predicates[rel[2]]
        print(sub_label, pred, obj_label)

    tree = RelationTree(gt_classes, classes)
    tree.gentree(trees)
    tree.drawTree()