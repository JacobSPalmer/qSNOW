from __future__ import annotations

from typing import Callable, Tuple, Literal, List, Optional

from stim import Circuit, GateTarget

from statistics import mean

from ..chip import LogicalTile
from ..models import Coord, TileTag, CSSType
from ..rules import ChannelRule, InjectionRule, Ruleset

class SCTile(LogicalTile):
    def __init__(self, 
                 distance: int, 
                 rounds: int,
                 task: Literal['memory_x', 'memory_z'] = 'memory_z',
                 origin: Coord = (0,0)):
        generator = lambda t, d, r: Circuit.generated(code_task=f"surface_code:rotated_{t}", distance=d, rounds=r)
        super().__init__(circuit = generator(task, distance, rounds).flattened(), 
                         initial_shift = None, 
                         x_buffer = 1, 
                         y_buffer = 1, 
                         origin = origin,
                         tag = TileTag(name=f"rsc_{task}_d{distance}",
                                       tile_type = type(self).__name__,
                                       distance = distance,
                                       rounds = rounds,
                                       generator=generator,
                                       generator_args={'code_task':f"surface_code:rotated_{task}", 'distance': distance, 'rounds': rounds}))
        
    #super hacky way to get typing and could probably be cleaned up but it should work
    def _init_tile_qubit_types(self):
        all_qubits = self._c2i
        all_measures = {}
        x_measures = {}
        z_measures = {}

        i2e = self._circuit.get_final_qubit_coordinates()
        for i in self._circuit:
            if i.name == "H":
                x_measures = {(i2e[q.value][0],i2e[q.value][1]):q.value for q in i.targets_copy()}
                break
        
        for c, i in all_qubits.items():
            if c[0]%2 == self.origin[0]%2:
                all_measures[c] = i

        z_measures = {k:all_measures[k] for k in all_measures.keys() - x_measures.keys()}


        self._indices = {
            'all_measures' 
            'x_measures'
            'z_measures'
            'data'
        }
        for q in self.qubits:
            if q.loc in x_measures:
                q.type = CSSType.X_CHECK
            elif q.loc in z_measures:
                q.type = CSSType.Z_CHECK
            elif q.loc in all_qubits:
                q.type = CSSType.DATA
            else:
                q.type = CSSType.BUFFER

    # TODO shift the custom_rules into the ChannelRuleset object, adding the add_rule method within the LogicalTile super class
    def _init_ruleset(self, custom_rules: List[InjectionRule] = []) -> Ruleset:

        def on_operation(op: str, trig, before, after) -> InjectionRule:
            return InjectionRule(op, trig, before, after)
        
        def apply_channel(channel, filter, scalar = 1.0):
            return ChannelRule(channel, filter, scalar = 1.0)

        return Ruleset([on_operation('R', 'all_qubits',
                                    before = [],
                                    after=[apply_channel('X_ERROR', 'all_qubits')]),
                        on_operation('H', 'x_measures',
                                    before = [],
                                    after=[apply_channel('DEPOLARIZE1', 'all_qubits')]),
                        on_operation('CX', 'any',
                                    before = [],
                                    after=[apply_channel('DEPOLARIZE2', 'active', 1.2),
                                            apply_channel('DEPOLARIZE1', 'idle')]),
                        on_operation('MR', 'all_measures',
                                    before=[apply_channel('X_ERROR', 'all_measures'),
                                            apply_channel('DEPOLARIZE1', 'data')],
                                    after=[apply_channel('X_ERROR', 'all_measures'),
                                            apply_channel('DEPOLARIZE1', 'data')]),
                        on_operation('M', 'data',
                                    before=[apply_channel('X_ERROR', 'all_qubits')],
                                    after = [])
                    ] + custom_rules)


    #TODO
    def copy(self) -> SCTile:
        return SCTile(distance=self.tag.distance, rounds=self.tag.rounds, task=self.tag.metadata['task']) # type: ignore