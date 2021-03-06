''' Batched Room-to-Room navigation environment '''

import sys


from nlp_spacy_nltk import string_cleaner_nlp, load_pathid_to_direction_idx
from nlp_spacy_nltk import remove_directions, contrafactual_directions, load_directions_and_contrafactual
from nlp_spacy_nltk import load_pathid_to_obj_idx, remove_object, replace_object, load_list_of_objs
sys.path.append('buildpy36')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args
#import copy


from utils import load_datasets, load_nav_graphs, Tokenizer

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class ObjEnvBatch(EnvBatch):
    def __init__(self, feature_store=None,obj_d_feat = None,obj_s_feat=None,batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        super(ObjEnvBatch, self).__init__(feature_store, batch_size)
        if obj_d_feat is not None:
           self.obj_d_feat = obj_d_feat
        else:
           self.obj_d_feat = None
        if obj_s_feat is not None:
            self.obj_s_feat = obj_s_feat
        else:
            self.obj_s_feat = None
        # self.obj_d_feat = obj_d_feat
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        obj_d_feat_states = []
        obj_s_feat_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
            if self.obj_d_feat:
                obj_d_feat = self.obj_d_feat[long_id]
                obj_d_feat_states.append(obj_d_feat)
            else:
                obj_d_feat_states.append(None)
            if self.obj_s_feat:
                obj_s_feat = self.obj_s_feat[long_id]
                obj_s_feat_states.append(obj_s_feat)
            else:
                obj_s_feat_states.append(None)
        return feature_states, obj_d_feat_states, obj_s_feat_states


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''
    # ESTE ES EL QUE CARGA ALL
    # SE GUARDA EN R2RBATCH.data que es una lista, y tiene un dict
    # con la data de las instrucciones y el encode

    def __init__(self, feature_store, obj_d_feat=None, obj_s_feat=None, batch_size=100, seed=10, splits=['train'],
                 tokenizer=None, name=None):
        if obj_d_feat or obj_s_feat:
            self.env = ObjEnvBatch(feature_store=feature_store, obj_d_feat=obj_d_feat, obj_s_feat=obj_s_feat,
                                   batch_size=batch_size)
        else:
            self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        #objs_certain, scanid_to_objs = load_scan_objs_data()# ADDED
        list_of_objs = load_list_of_objs()
        pathid_to_direction_idx = load_pathid_to_direction_idx() 
        directions_and_contrafactual = load_directions_and_contrafactual()
        pathid_to_obj_idx = load_pathid_to_obj_idx()
        for split in splits:
            for item in load_datasets([split]):
                pathid = item["path_id"]
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    #print(instr)
                    #print(new_item['instr_id'])
                    #print("####")
                    instr = string_cleaner_nlp(instr)
                    if split == "train":
                        if args.no_object:
                            instr = remove_object(pathid_to_obj_idx, instr,j,pathid)
                        elif args.no_text:
                            instr = " ".join(["<UNK>" for word in instr.split(' ')])
                        elif args.no_directions:
                            instr = remove_directions(pathid_to_direction_idx, instr,j,pathid)
                        elif args.contrafactual_directions:
                            instr = contrafactual_directions(pathid_to_direction_idx, instr,j,pathid, directions_and_contrafactual)
                        elif args.replace_object:
                            #print(instr)
                            instr = replace_object(pathid_to_obj_idx, instr,j,pathid, list_of_objs)
                            #print(instr)
                            #print("#####")
                    new_item['instructions'] = instr
                    new_item["instructions_idx"] = j
                    #print("vanilla instr type =", instr)
                    #copy_instr = copy.copy(instr)
                    #if item["scan"] in scanid_to_objs:
                    #    fake_instr = swap_objs_using_scanid(objs_certain, scanid_to_objs, item['scan'], copy_instr, alpha=1)
                        #print("fake instr type =",fake_instr)

                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                       
                        #print(instr, new_item['instr_encoding'])
                        #if item["scan"] in scanid_to_objs:
                        #    new_item['fake_instr_encoding'] = tokenizer.encode_sentence(fake_instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.angle_avg_feature = utils.get_avg_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        # for i, (feature, state) in enumerate(self.env.getStates()):
        # print(self.env)
        # print(len(self.env.getStates()))
        if args.sparseObj or args.denseObj:
            try:
                F, obj_d_feat, obj_s_feat = self.env.getStates()
            except Exception as e:
                print(str(e)) 
                import pdb; pdb.set_trace()
        else:
            F = self.env.getStates()
        for i in range(len(F)):
            feature = F[i][0]
            state = F[i][1]
            # odf = obj_d_feat[i]
            if args.sparseObj:
                osf = obj_s_feat[i]
            if args.denseObj:
                odf = obj_d_feat[i]
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs_dict = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id'],
                'instr_idx' : item['instructions_idx']
            }
            if args.sparseObj:
                if args.catfeat == 'none':
                    obs_dict['obj_s_feature'] = osf['concat_feature']
                elif args.denseObj:
                    obs_dict['obj_s_feature'] = osf['concat_feature']
                elif args.catfeat == 'he': # avoid cat angle 2 times
                    if osf['concat_text'][0] == 'zero':
                        obs_dict['obj_s_feature'] = np.concatenate(
                            (osf['concat_feature'], np.zeros((1, 16))), axis=1)
                    elif osf['concat_text'][0] == 'average':
                        he = osf['concat_angles']
                        obs_dict['obj_s_feature'] = np.concatenate(
                            (osf['concat_feature'], he), 1)
                    else:
                        obs_dict['obj_s_feature'] = np.zeros(
                            (len(osf['concat_feature']), 16 + 300))
                        for k, v in enumerate(osf['concat_viewIndex']):
                            # he = np.tile(np.concatenate((odf['concat_angles_h'][k], odf['concat_angles_e'][k])),
                            #              args.angle_feat_size // 8)
                            he = osf['concat_angles'][k]
                            obs_dict['obj_s_feature'][k] = np.concatenate(
                                (osf['concat_feature'][k], he))
                elif args.catfeat == 'angle':
                    if osf['concat_text'][0] == 'zero':
                        obs_dict['obj_s_feature'] = np.concatenate(
                            (osf['concat_feature'], np.zeros((1,args.instHE))), axis=1)
                    elif osf['concat_text'][0] == 'average':
                        obs_dict['obj_s_feature'] = np.concatenate(
                            (osf['concat_feature'], np.expand_dims(self.angle_avg_feature[base_view_id][:args.instHE]
                                                                   ,axis=0)),axis=1)
                    else:
                        obs_dict['obj_s_feature'] = np.zeros(
                            (len(osf['concat_feature']), args.instEmb+args.instHE))
                        for k, v in enumerate(osf['concat_viewIndex']):
                            obs_dict['obj_s_feature'][k] = np.concatenate(
                                (osf['concat_feature'][k], self.angle_feature[base_view_id][v][:args.instHE]))
            if args.denseObj:
                if args.catfeat == 'none':
                    obs_dict['obj_d_feature'] = odf['concat_feature']
                elif args.catfeat =='bboxAngle': # concat bbox 2d coordinates & vision_angle_feature
                    if odf['concat_text'][0] == 'zero':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], np.zeros((1,args.angle_feat_size))),axis=1)
                    elif odf['concat_text'][0] == 'average':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], np.tile(odf['concat_bbox'], (args.angle_feat_size/2)//4),
                             np.expand_dims(self.angle_avg_feature[base_view_id][:args.angle_feat_size/2], axis=0)), axis=1)
                    else:
                        obs_dict['obj_d_feature'] = np.zeros((len(odf['concat_feature']), args.angle_feat_size
                                                              +args.feature_size))
                        for k,v in enumerate(odf['concat_viewIndex']):
                            obs_dict['obj_d_feature'][k] = np.concatenate(
                                (odf['concat_feature'][k], np.tile(odf['concat_bbox'][k], (args.angle_feat_size/2)//4),
                                 self.angle_feature[base_view_id][v][:args.angle_feat_size/2]))
                elif args.catfeat == 'bbox': # concat bbox 2d coordinates
                    if odf['concat_text'][0] == 'zero':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], np.zeros((1,args.angle_feat_size))), axis=1)
                    elif odf['concat_text'][0] == 'average':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'],np.tile(odf['concat_bbox'], args.angle_feat_size//4)),axis=1)
                    else:
                        obs_dict['obj_d_feature'] = np.zeros(
                            (len(odf['concat_feature']), args.angle_feat_size + args.feature_size))
                        for k, v in enumerate(odf['concat_viewIndex']):
                            obs_dict['obj_d_feature'][k] = np.concatenate(
                                (odf['concat_feature'][k], np.tile(odf['concat_bbox'][k], args.angle_feat_size // 4)))
                elif args.catfeat == 'angle': # concat vision_angle_feature
                    if odf['concat_text'][0] == 'zero':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], np.zeros((1, args.angle_feat_size))), axis=1)
                    elif odf['concat_text'][0] == 'average':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], np.expand_dims(self.angle_avg_feature[base_view_id], axis=0)),
                            axis=1)
                    else:
                        obs_dict['obj_d_feature'] = np.zeros(
                            (len(odf['concat_feature']), args.angle_feat_size + args.feature_size))
                        for k, v in enumerate(odf['concat_viewIndex']):
                            obs_dict['obj_d_feature'][k] = np.concatenate(
                                (odf['concat_feature'][k], self.angle_feature[base_view_id][v]))
                elif args.catfeat == 'he': # concat bbox2angle
                    if odf['concat_text'][0] == 'zero':
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], np.zeros((1, args.angle_feat_size))), axis=1)
                    elif odf['concat_text'][0] == 'average':
                        # he = np.tile(np.concatenate((odf['concat_angles_h'],odf['concat_angles_e']),axis=1),args.angle_feat_size//8)
                        he = odf['concat_angles']
                        obs_dict['obj_d_feature'] = np.concatenate(
                            (odf['concat_feature'], he), 1)
                    else:
                        obs_dict['obj_d_feature'] = np.zeros(
                            (len(odf['concat_feature']), args.angle_feat_size + args.feature_size))
                        for k, v in enumerate(odf['concat_viewIndex']):
                            # he = np.tile(np.concatenate((odf['concat_angles_h'][k], odf['concat_angles_e'][k])),
                            #              args.angle_feat_size // 8)
                            he = odf['concat_angles'][k]
                            obs_dict['obj_d_feature'][k] = np.concatenate(
                                (odf['concat_feature'][k], he))
            obs.append(obs_dict)
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            #if 'fake_instr_encoding' in item:
            #    obs[-1]['fake_instr_encoding'] = item['fake_instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


