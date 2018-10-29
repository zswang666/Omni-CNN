import os, shutil
import numpy as np
import pdb
import cv2
import subprocess
import pickle

def run_sfm(scene_name, path_list):
    """
    This is the function to connect to opensfm
    given scene name and list of full paths, return a dict of string to tuple of np.1darray
    the key of dict is string (image path) and value is tuple of np.1darray (np.array([x,y,z]), np.array([ax,ay,az]))
    NOTE!: in order to not run opensfm for same scene over one time, please save the sfm result for every scene using scene tag
            when user try to sfm the same scene again, ask them whether to rerun opensfm or use old result to save time
    Input:
        scene_name: string; the name of the scene, it is also the unique tag for the scene
        path_list: list of string; list of full path of the input images to opensfm, the images are all W=1280, H=768, D=3
    Return:
        image_dict: string to tuple of np.1darray;
            keys: image_path: string; the given path
            values: image_location: tuple of np.1darray; the location (position, rotation) information of the image
    """
    check = 'N'
    root_dir = "/home/henry/OpenSfM/data/%s"%scene_name
    from_dir_r = "/home/henry/OpenSfM/rotation.pickle"
    from_dir_t = "/home/henry/OpenSfM/translation.pickle"
    directory = "/home/henry/OpenSfM/data/%s"%scene_name
    imagedes = "/home/henry/OpenSfM/data/%s/images"%scene_name
    config = "/home/henry/OpenSfM/config.yaml"
    arg = ["/home/henry/OpenSfM/bin/opensfm_run_all", "data/%s"%scene_name]
    if not os.path.exists(directory):
        os.makedirs(directory) # create all directories, raise an error if it already exists
        os.makedirs(imagedes)
        shutil.copy(config, directory)
        for index, imagefrom in enumerate(path_list):
            src_dir = imagefrom
            dst_dir = imagedes
            #dst_image =  os.path.join(dst_dir, os.path.dirname(src_dir))
            #os.makedirs(dst_dir)
            shutil.copy(src_dir, dst_dir)
            change = os.path.basename(src_dir)#############################
            newname = src_dir.replace("/", " ")
            os.rename(dst_dir+"/"+change, dst_dir+"/"+src_dir)#############
        print "Path_list ------------> ", path_list
        subprocess.call(arg) #run opensfm here
        shutil.copy(from_dir_r, root_dir)
        shutil.copy(from_dir_t, root_dir)

        #"read trans and rota here"            
        fr = None
        try:
            fr = open(os.path.join(root_dir, 'rotation.pickle'), 'r')
            rotation = pickle.load(fr)
        except (EnvironmentError, pickle.PicklingError) as err:
            raise ValueError(str(err))
        finally:
            if fr is not None:
                fr.close()
        ft = None
        try:
            ft = open(os.path.join(root_dir, 'translation.pickle'), 'r')
            translation = pickle.load(ft)
        except (EnvironmentError, pickle.PicklingError) as err:
            raise ValueError(str(err))
        finally:
            if ft is not None:
                ft.close()
        translation.pop('example', None)
        rotation.pop('example', None)
        '''
        # change Y Z
        global rtemp1
        global rtemp2
        global ttemp1
        global ttemp2
        for x,rota in enumerate(rotation):
        for n,i in enumerate(rotation):
            if n==1 :
                rtemp1 = rotation[n]
            if n==2 :
                rtemp2 = rotation[n]
        for n,i in enumerate(rotation):
            if n==1 :
                rotation[n] = rtemp2
            if n==2 :
                rotation[n] = rtemp1
        for n,i in enumerate(translation):
            if n==1 :
                ttemp1 = translation[n]
            if n==2 :
                ttemp2 = translation[n]
        for n,i in enumerate(translation):
            if n==1 :
                translation[n] = ttemp2
            if n==2 :
                translation[n] = ttemp1
        '''
        c = {}
        for image_path in rotation.keys():
            c[image_path] = (translation[image_path],rotation[image_path])
        c = dict([(str(k), v) for k, v in c.items()])
    else:
        print "Do you want to reconstruct again? It may take you some times!!!"
        check = raw_input(">>>[Y/N]")
        if check == 'Y':
            #delet all doucment in images file and itself
            shutil.rmtree(imagedes)
            os.makedirs(imagedes)
            #store images in it
            for index, imagefrom in enumerate(path_list):
                src_dir = imagefrom
                dst_dir = imagedes
                #dst_dir =  os.path.join(dst_dir, os.path.dirname(src_dir))
                #os.makedirs(dst_dir)
                shutil.copy(src_dir, dst_dir)
                change = os.path.basename(src_dir)
                newname = src_dir.replace("/", " ")
                os.rename(dst_dir+"/"+change, dst_dir+"/"+newname)
            print "PATH==================>", path_list
            pdb.set_trace()
            subprocess.call(arg) #run opensfm here
            shutil.copy(from_dir_r, root_dir)
            shutil.copy(from_dir_t, root_dir)
            #"read trans and rota here"            
            fr = None
            try:
                fr = open(os.path.join(root_dir, 'rotation.pickle'), 'r')
                rotation = pickle.load(fr)
            except (EnvironmentError, pickle.PicklingError) as err:
                raise ValueError(str(err))
            finally:
                if fr is not None:
                    fr.close()
            ft = None
            try:
                ft = open(os.path.join(root_dir, 'translation.pickle'), 'r')
                translation = pickle.load(ft)
            except (EnvironmentError, pickle.PicklingError) as err:
                raise ValueError(str(err))
            finally:
                if ft is not None:
                    ft.close()
            translation.pop('example', None)
            rotation.pop('example', None)
            '''
            # change Y Z
            global rtemp1
            global rtemp2
            global ttemp1
            global ttemp2
            for n,i in enumerate(rotation):
                if n==1 :
                    rtemp1 = rotation[n]
                if n==2 :
                    rtemp2 = rotation[n]
            for n,i in enumerate(rotation):
                if n==1 :
                    rotation[n] = rtemp2
                if n==2 :
                    rotation[n] = rtemp1
            for n,i in enumerate(translation):
                if n==1 :
                    ttemp1 = translation[n]
                if n==2 :
                    ttemp2 = translation[n]
            for n,i in enumerate(translation):
                if n==1 :
                    translation[n] = ttemp2
                if n==2 :
                    translation[n] = ttemp1
            '''
            #r and t become         
            c = {}
            for image_path in rotation.keys():
                c[image_path] = (translation[image_path],rotation[image_path])
            c = dict([(str(k), v) for k, v in c.items()]) 
        else:
            #"read trans and rota here"            
            fr = None
            try:
                fr = open(os.path.join(root_dir, 'rotation.pickle'), 'r')
                rotation = pickle.load(fr)
            except (EnvironmentError, pickle.PicklingError) as err:
                raise ValueError(str(err))
            finally:
                if fr is not None:
                    fr.close()

            ft = None
            try:
                ft = open(os.path.join(root_dir, 'translation.pickle'), 'r')
                translation = pickle.load(ft)
            except (EnvironmentError, pickle.PicklingError) as err:
                raise ValueError(str(err))
            finally:
                if ft is not None:
                    ft.close()



            translation.pop('example', None)
            rotation.pop('example', None)
            '''
            # change Y Z
            global rtemp1
            global rtemp2
            global ttemp1
            global ttemp2
            for n,i in enumerate(rotation):
                if n==1 :
                    rtemp1 = rotation[n]
                if n==2 :
                    rtemp2 = rotation[n]
            for n,i in enumerate(rotation):
                if n==1 :
                    rotation[n] = rtemp2
                if n==2 :
                    rotation[n] = rtemp1
            for n,i in enumerate(translation):
                if n==1 :
                    ttemp1 = translation[n]
                if n==2 :
                    ttemp2 = translation[n]
            for n,i in enumerate(translation):
                if n==1 :
                    translation[n] = ttemp2
                if n==2 :
                    translation[n] = ttemp1
            '''
            c = {}
            for image_path in rotation.keys():
                c[image_path] = (translation[image_path],rotation[image_path])
            c = dict([(str(k), v) for k, v in c.items()])
    print c
    return c
    raise NotImplementedError("not implemented")
