from torch.utils.tensorboard import SummaryWriter
from microrts.rts_wrapper.envs.utils import encoded_utt_dict
from microrts.algo.model import ActorCritic
from torch import optim
import torch
from microrts.sl.sl_data_processor import get_data
import microrts.settings as settings
import os
"""
<state, player, units actions assignment>
1. storage <state, player, unit, action>  
2. sample: encode as {unitType : (state, unit, action features, numpy)}
3. (spatial_features, unit_features, actions_label) unit_features: units cat utt
4. forward according to unit_types -> backward -> forward ...
"""


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    storage = get_data(saving_dir=os.path.join(settings.data_dir, "rvr6x6.pck"))
    model = ActorCritic(map_size=(6, 6))
    # model = ActorCritic(map_size=(8, 8))
    writer = SummaryWriter()

    # input()
    model.to(device)

    iteration = int(1e6)
    batch_size = 128
    criteria = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-6)

    for i in range(iteration):
        loss = 0
        sample_dict = storage.sample(batch_size)
        for key in sample_dict:
            if key not in model.activated_agents:
                continue
            if sample_dict[key]:
                # (states, units, units_actions)
                spatial_features, unit_features, actions = sample_dict[key]

                spatial_features = torch.from_numpy(spatial_features).float().to(device)
                unit_features = torch.from_numpy(unit_features).float().to(device)

                encoded_utt = torch.from_numpy(encoded_utt_dict[key]).unsqueeze(0).float().repeat(unit_features.size(0), 1).to(device)
                # cat utt and the individual feature together
                unit_features = torch.cat([unit_features, encoded_utt], dim=1)
                actions = torch.from_numpy(actions).long().to(device)
                # print(states.device, units.device)
                probs = model.actor_forward(key, spatial_features, unit_features)
                # print(probs.device)
                # input()
                # _actions = torch.zeros_like(prob)
                # for i in range(len(actions)):
                #     _actions[i][actions[i]] = 1

                log_probs = torch.log(probs)
                loss += criteria(log_probs, actions)
        if i % 100 == 0:
            writer.add_scalar("all losses", loss, i)
            print("iter{}, loss:{}".format(i, loss))

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        optimizer.step()
        # print(prob[i])

    torch.save(model.state_dict(), os.path.join(settings.microrts_path, "models", "1M.pth"))


if __name__ == '__main__':
    main()