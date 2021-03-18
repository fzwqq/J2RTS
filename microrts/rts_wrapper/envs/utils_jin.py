import socket

from numpy.core._multiarray_umath import ndarray
from .utils import encoded_utt_feature_size, utt_feature
from .datatypes import *
import json
import numpy as np
import time
import torch
from torch.distributions import Categorical


def signal_wrapper_v2(raw):
    """wrap useful signal from java raw

        Arguments:
            raw {str} -- msg received from java

        Returns:
            tuple of --
            observation {np.array}
            reward {float}
            done {bool}
            info {dict}
    """
    # print(raw)
    curr_player = int(raw.split('\n')[0].split()[1])
    gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(raw.split('\n')[1]))
    pgs = gs_wrapper.gs.pgs
    # print("raw datas from java are:")
    # print(raw)
    reward = gs_wrapper.reward
    done = gs_wrapper.done
    observation = state_encoder_v2(gs_wrapper.gs, curr_player)
    # self.game_time = gs_wrapper.gs.time
    info = {
        "unit_valid_actions": gs_wrapper.validActions,  # friends and their valid actions
        # "units_list": [u for u in gs_wrapper.gs.pgs.units],
        # "enemies_actions": None,
        "resources": pgs.players[curr_player].resources,
        "units_on_field": [u.ID for u in gs_wrapper.gs.pgs.units],
        "winner": gs_wrapper.winner,
        "current_player": curr_player,
        "player_resources": [p.resources for p in gs_wrapper.gs.pgs.players],
        "map_size": [gs_wrapper.gs.pgs.height, gs_wrapper.gs.pgs.width],
        "time_stamp": int(gs_wrapper.gs.time),
    }
    return observation, reward, done, info


def hp_ration_encoder(hp_ratio: float):
    """
    one hot encoding of hp ratio range:
    None,
    0% ~ 10%,
    10% ~ 20%,
    20% ~ 40%,
    40% ~ 80%,
    80% ~ 100%
    Arguments:
        hp_ratio{float} -- The given hp ration
    Return:
        position in ont hot encoding
    """
    if hp_ratio == 0:
        return 0
    elif hp_ratio < .1:
        return 1
    elif hp_ratio < .2:
        return 2
    elif hp_ratio < .4:
        return 3
    elif hp_ratio < .8:
        return 4
    else:
        return 5


def resource_encoder(amount, feature_length=8, amout_threshold=2):
    """
    0, 1, 2, 4, 8, 16, 32, other
    Arguments:
        amount {[type]} -- [description]

    Keyword Arguments:
        feature_length {int} -- [description] (default: {8})
        amount_threshold {int} -- [description] (default: {2})

    Returns:
        [type] -- [description]
    """
    resource = np.zeros(8)
    if amount == 0:
        resource[0] = 1
        return resource
    bit_pos = 1
    for _ in range(1, feature_length):
        if amount > amout_threshold:
            bit_pos += 1
            amout_threshold *= 2
        else:
            resource[bit_pos] = 1
            return resource
    resource[-1] = 1
    return resource


def game_action_translator_v2(u: Unit, ua: UnitAction):
    """
    translate the game actions to ones network readable

    Return:
        network action index {list}
            list[0] -- [action_type_idx]
            list[1] -- [direction_type_idx]
            list[2] -- [produce_type_idx]
    """
    nna_idx = [0, 0, 0]

    idx_action_type = 0  # 5 + 1
    idx_action_parameter = 1  # 4 + 1
    idx_produce_type = 2  # 6 + 1

    def attack_trans(x, y, _x, _y):
        for i in range(4):
            if (x + DIRECTION_OFFSET_X[i] == _x) and (y + DIRECTION_OFFSET_Y[i] == _y):
                return i

    def action_unit_type_idx(ua_unit_type):
        if ua_unit_type == "":
            nna_idx[idx_produce_type] = -1
        elif ua_unit_type == UNIT_TYPE_NAME_BASE:
            nna_idx[idx_produce_type] = 0
        elif ua_unit_type == UNIT_TYPE_NAME_BARRACKS:
            nna_idx[idx_produce_type] = 1
        elif ua_unit_type == UNIT_TYPE_NAME_WORKER:
            nna_idx[idx_produce_type] = 2
        elif ua_unit_type == UNIT_TYPE_NAME_HEAVY:
            nna_idx[idx_produce_type] = 3
        elif ua_unit_type == UNIT_TYPE_NAME_LIGHT:
            nna_idx[idx_produce_type] = 4
        elif ua_unit_type == UNIT_TYPE_NAME_RANGED:
            nna_idx[idx_produce_type] = 5

    if ua.type == ACTION_TYPE_ATTACK_LOCATION:
        attack_trans(u.x, u.y, ua.x, ua.y)
    else:
        if ua.parameter >= 4:
            nna_idx[idx_action_parameter] = ACTION_TYPE_NONE
        else:
            nna_idx[idx_action_parameter] = ua.parameter

    nna_idx[idx_action_type] = ua.type
    action_unit_type_idx(ua.unitType)

    return nna_idx


def state_encoder_v2(gs: GameState, player):
    """Encode the state for player given Game state from java

    Arguments:
        gs {GameState} -- Game state from java
        player {int} -- current player

    Returns:
        np.array -- features
    """
    current_player = player
    pgs = gs.pgs
    w = pgs.width
    h = pgs.height
    units = pgs.units

    p1_info, p2_info = gs.pgs.players
    my_resources = p1_info.resources if current_player == p1_info.ID else p2_info.resources
    opp_resources = p2_info.resources if current_player == p1_info.ID else p1_info.resources

    cannot_walk = []
    can_walk = []
    for x in pgs.terrain:
        if int(x) == 0:
            cannot_walk.append(0)
            can_walk.append(1)
        else:
            cannot_walk.append(1)
            can_walk.append(0)
    channel_whether_walkable = np.array([cannot_walk, can_walk]).reshape((2, h, w))
    channel_resource_size = np.zeros((8, h, w))
    channel_units_type = np.zeros((len(UNIT_COLLECTION), h, w))
    channel_whether_mine = np.zeros((2, h, w))

    channel_player = np.zeros((2, h, w))
    channel_player[current_player, :, :] = 1

    channel_hp_ration = np.zeros((6, h, w))

    channel_my_resources = np.zeros((8, h, w))
    channel_opp_resources = np.zeros((8, h, w))
    _one_hot_my_resource_pos = list(resource_encoder(my_resources)).index(1)
    _one_hot_opp_resource_pos = list(resource_encoder(opp_resources)).index(1)
    channel_my_resources[_one_hot_my_resource_pos, :, :] = 1
    channel_my_resources[_one_hot_opp_resource_pos, :, :] = 1

    id_location_map = {}
    for unit in units:
        _owner = unit.player
        _type = unit.type
        _x, _y = unit.x, unit.y
        _resource_carried = unit.resources
        _hp = unit.hitpoints
        _id = unit.ID

        _one_hot_hp_ratio_pos = hp_ration_encoder(_hp / UTT_DICT[_type].hp)
        channel_hp_ration[_one_hot_hp_ratio_pos][_x][_y] = 1

        _one_hot_resource_pos = list(resource_encoder(_resource_carried)).index(1)
        channel_resource_size[_one_hot_resource_pos][_x][_y] = 1

        _one_hot_type_pos = UNIT_COLLECTION.index(_type)
        channel_units_type[_one_hot_type_pos][_x][_y] = 1

        if _owner == current_player:
            channel_whether_mine[1][_x][_y] = 1
            channel_whether_mine[0][_x][_y] = 0
        else:
            channel_whether_mine[1][_x][_y] = 0
            channel_whether_mine[0][_x][_y] = 1

        id_location_map[_id] = unit

        # 5 + 6 + 7
    channel_action_parm = np.zeros((5, h, w))
    channel_action_type = np.zeros((6, h, w))
    channel_action_prod = np.zeros((7, h, w))

    for uaa in gs.actions:
        _id = uaa.ID
        _action = uaa.action
        _unit = id_location_map[_id]
        _x, _y = _unit.x, _unit.y
        _one_hot_action_type_pos, _one_hot_action_parm_pos, \
            _one_hot_action_prod_type = game_action_translator_v2(_unit, _action)
        channel_action_type[_one_hot_action_type_pos][_x][_y] = 1
        channel_action_parm[_one_hot_action_parm_pos][_x][_y] = 1
        channel_action_prod[_one_hot_action_prod_type][_x][_y] = 1

    spatial_features: ndarray = np.vstack(
        (
            channel_whether_walkable,
            channel_resource_size,
            channel_hp_ration,
            channel_units_type,

            channel_whether_mine,
            channel_my_resources,
            channel_opp_resources,
            channel_player,

            channel_action_type,
            channel_action_parm,
            channel_action_prod
        ),
    )
    return spatial_features


# reward is related to the player actions but not single unit action,
# that is it 's better to cal the global reward
# and then use the global reward to update the policy
# even though it's hard to define the action is good or not
# with this situation, we could do back pro separately


def unit_feature_encoder_v2(unit: Unit, map_size: list):
    map_height, map_width = map_size
    owner = unit.player
    unit_type = unit.type
    unit_x = unit.x
    unit_y = unit.y
    unit_resource = unit.resources
    unit_hp = unit.hitpoints

    owner_feature = np.zeros(2)
    owner_feature[owner] = 1

    type_feature = np.zeros(len(UNIT_COLLECTION))
    type_feature[UNIT_COLLECTION.index(unit_type)] = 1

    x_ratio_feature = np.array([unit_x / map_width])
    y_ratio_feature = np.array([unit_y / map_height])

    resource_feature = resource_encoder(unit_resource)

    hp_ratio_feature = np.zeros((6,))
    hp_ratio_feature[hp_ration_encoder(unit_hp / UTT_DICT[unit_type].hp)] = 1

    is_carry = np.zeros(2)
    unit_feature = np.hstack(
        (
            owner_feature,  # 2
            type_feature,  # 7
            hp_ratio_feature,  # 6
            resource_feature,  # 8
            x_ratio_feature,  # 1
            y_ratio_feature,  # 1
        )
    )
    return unit_feature


def action_sampler_v3(model, state, info, device='cpu', mode='stochastic', hidden_states: dict = None, callback=None,
                      debug=False):
    assert mode in ['stochastic', 'deterministic']
    unit_valid_actions = info["unit_valid_actions"]
    map_size = info["map_size"]

    states = []
    units = []
    hxses = []
    utts = []
    samples = []

    if unit_valid_actions:
        for uva in unit_valid_actions:
            u = uva.unit
            states.append(state)
            units.append(unit_feature_encoder_v2(u, map_size))
            utts.append(utt_feature)
            if model.recurrent:
                if u.ID in hidden_states.keys():  # check if hidden_states has hidden state of the unit, otherwise, init to zero
                    hxses.append(hidden_states[u.ID])
                else:
                    hxses.append(np.zeros((256)))
        batch = (np.array(states), np.array(units), np.array(utts), np.array(hxses))

        with torch.no_grad():
            states, units, utts, hxses = batch
            states = torch.from_numpy(states).float().to(device)
            units = torch.from_numpy(units).float().to(device)
            utts = torch.from_numpy(utts).float().to(device)
            hxses = torch.from_numpy(hxses).float().to(device)
            value, policy, hxs = model.forward(spatial_feature=states,
                                               unit_feature=units,
                                               utt_feature=utts,
                                               hxses=hxses.unsqueeze(0))
            actions_type = []
            actions_parm = []
            actions_prod = []

            if mode == "stochastic":
                ua_type = Categorical(policy[0][:])
                ua_idxes_type = ua_type.sample()
                ua_parm = Categorical(policy[1][:])
                ua_idxes_parm = ua_parm.sample()
                ua_prod = Categorical(policy[2][:])
                ua_idxes_prod = ua_prod.sample()

                for ua_idx_type, ua_idx_parm, ua_idx_prod in zip(ua_idxes_type, ua_idxes_parm, ua_idxes_prod):
                    actions_type.append(ua_idx_type)
                    actions_parm.append(ua_idx_parm)
                    actions_prod.append(ua_idx_prod)
            elif mode == "deterministic":
                ua_idxes_type = torch.max(policy[0][:], 1)
                ua_idxes_parm = torch.max(policy[1][:], 1)
                ua_idxes_prod = torch.max(policy[2][:], 1)
                for ua_idx_type, ua_idx_parm, ua_idx_prod in zip(ua_idxes_type.index, ua_idxes_parm.indices,
                                                                 ua_idxes_prod.indices):
                    actions_type.append(ua_idx_type)
                    actions_parm.append(ua_idx_parm)
                    actions_prod.append(ua_idx_prod)
            samples = list(zip(unit_valid_actions, actions_type, actions_parm, actions_prod))
    return samples, hxses


def normalize(value, max_v, min_v):
    return 1 if max_v == min_v else (value - min_v) / (max_v - min_v)


def network_action_translator_v2(unit_validaction_choices) -> List[PlayerAction]:
    """
    translate network actions to ones game readable
    :param unit_validaction_choices: tuple of ([unit with valid action list], [unitAction instance form datatypes])
    :return:
    """
    pas = []
    for uva, ua_type, ua_parm, ua_prod in unit_validaction_choices:
        unit = uva.unit
        valid_actions = uva.unitActions
        pa = PlayerAction(unitID=unit.ID)  # unit action assignment
        for action in valid_actions:
            if action.type == ua_type:
                if ua_type == ACTION_TYPE_NONE:
                    continue
                elif ua_type == ACTION_TYPE_MOVE and action.parameter == ua_parm:
                    pa.unitAction = action
                elif ua_type == ACTION_TYPE_RETURN and action.parameter == ua_parm:
                    pa.unitAction = action
                elif ua_type == ACTION_TYPE_HARVEST and action.parameter == ua_parm:
                    pa.unitAction = action
                elif ua_type == ACTION_TYPE_ATTACK_LOCATION:
                    if action.x == unit.x + DIRECTION_OFFSET_X[ua_parm] \
                            and action.y == unit.y + DIRECTION_OFFSET_Y[ua_parm]:
                        pa.unitAction = action
                elif ua_type == ACTION_TYPE_PRODUCE:
                    if action.unitType == AGENT_COLLECTION[ua_prod] and action.parameter == ua_parm:
                        pa.unitAction = action
        pas.append(pa)
    return pas
