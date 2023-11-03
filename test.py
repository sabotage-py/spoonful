import os
import shutil
import ipyplot
import random
from PIL import Image, ImageFile

from stage1 import get_image_class
from stage2 import cluster_similar_images, get_candidate

ImageFile.LOAD_TRUNCATED_IMAGES = True


def for_each_folder(path):
    """returns the indices of the selected
    FRONT and BACK images in the list os.listdir(path)
    """
    red = os.listdir(path)
    back, front, other = [], [], []
    output = {
        text: None for text in [
            "all", "front", "back", "front_candidates", 
            "back_candidates"
        ]
    }
    all_images = []
    for i in range(len(red)):
        try:
            img = Image.open(
                os.path.join(path, red[i])
            ).convert('RGB')
        except Exception as e:
            print(e)
            continue
        try:
            # image_classes = ['back', 'front', 'other']
            predicted_class = get_image_class(img)
            if predicted_class == 0:
                back.append(i)
            elif predicted_class == 1:
                front.append(i)
            else:
                other.append(i)
        except Exception as e:
            print(e)
            continue
        all_images.append((i, predicted_class))
    if all_images:
        output["all"] = all_images[:]
    if front:
        temp_path = os.path.join(path, 'temp_front')
        os.mkdir(temp_path)
        for i in front:
            shutil.copy(
                os.path.join(path, red[i]),
                os.path.join(temp_path, red[i])
            )
        front_cluster = cluster_similar_images(temp_path)
        if front_cluster is not None:
            temp_path_list = os.listdir(temp_path)
            output["front_candidates"] = [
                red.index(temp_path_list[idx]) for idx in front_cluster
            ]
            selected = front_cluster[0]
            output["front"] = [red.index(temp_path_list[selected])]
        shutil.rmtree(temp_path)
        if front_cluster is None:
            front_candidates = get_candidate(
                path, front, label='front'
            )
            if front_candidates is not None:
                output["front_candidates"] = front_candidates[:]
                selected = front_candidates[0]
                output["front"] = [selected]
    if back:
        back_candidates = get_candidate(
            path, back, label='back'
        )
        if back_candidates is not None:
            output["back_candidates"] = back_candidates[:]
            selected = back_candidates[0]
            output["back"] = [selected]
    return output


## --------------------------------------------------------------------
## the following functions were used in the DEMO and can be removed.
def get_metrics(folder, label='front'):
    path = 'F:/e2e_precision'
    folder_path = os.path.join(path, folder)
    output = for_each_folder(folder_path)
    red = os.listdir(folder_path)
    is_label = False
    for i in range(len(red)):
        if label in red[i]:
            is_label = True
            break
    tp, fp, fn = 0, 0, 0
    if output[label]:
        itr = output[label][0]
        if label in red[itr]:
            tp = 1
        else:
            fp = 1
            if is_label:
                fn = 1
    elif is_label:
        fn = 1
    return tp, fp, fn


def demo(folder='random', show_images=True, teaser=2):
    image_classes = ['back', 'front', 'other']
    path = 'F:/datadump/dataset9'
    folders = os.listdir(path)
    if type(folder) == type(1):
        folder = folders[folder]
    if folder == 'random':
        folder = random.choice(folders)
        print('Randomly chosen folder:', folder)
    if folder not in folders:
        print('ERROR: folder {} not found.'.format(folder))
        return None
    folder_path = os.path.join(path, folder)
    red = os.listdir(folder_path)
    output = for_each_folder(folder_path)
    return_value = [0, 0, 0]
    if output['all']:
        all_images = [
            Image.open(os.path.join(folder_path, red[tup[0]])) 
            for tup in output['all']
        ]
        if show_images:
            if teaser == 0:
                ipyplot.plot_images(all_images)
            else:
                ipyplot.plot_images(all_images, 
                                    custom_texts=[
                                        image_classes[tup[1]]
                                        for tup in output['all']
                                    ])
        return_value[0] = len(output['all'])
    else:
        print('The folder is either empty', 
              'or there is a problem with the image files')
        return return_value
    if teaser <= 1:
        return None
    if output['front']:
        front_image = [
            Image.open(os.path.join(folder_path, red[i])) 
            for i in output['front']
        ]
        if show_images:
            ipyplot.plot_images(front_image, ['front'], img_width=300)
        return_value[1] = 1
    else:
        print('NO FRONT CANDIDATE FOUND.')
        # if output['front_candidates']:
        #     print('However, these images were classified', 
        #           'as FRONT by the model.')
        #     front_images = [
        #         Image.open(os.path.join(folder_path, red[i])) 
        #         for i in output['front_candidates']
        #     ]
        #     if show_images:
        #         ipyplot.plot_images(front_images)
        # else:
        #     print('No images were classified as FRONT by the model.')
    if output['back']:
        back_image = [
            Image.open(os.path.join(folder_path, red[i])) 
            for i in output['back']
        ]
        if show_images:
            ipyplot.plot_images(back_image, ['back'], img_width=300)
        return_value[2] = 1
    else:
        print('NO BACK CANDIDATE FOUND')
        # if output['back_candidates']:
        #     back_images = [
        #         Image.open(os.path.join(folder_path, red[i]))
        #         for i in output['back_candidates']
        #     ]
        #     if show_images:
        #         ipyplot.plot_images(back_images)
        # else:
        #     print('No images were classified as BACK by the model.')
    return return_value
