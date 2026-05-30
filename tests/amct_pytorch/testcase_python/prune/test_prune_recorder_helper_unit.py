#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import logging
import unittest

from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pytorch_pb2 as pb,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.prune import (
    prune_recorder_helper as prh,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils.prune_record_attr_util import (
    AttrProtoHelper,
)

logger = logging.getLogger(__name__)

PRH = prh.PruneRecordHelper
REC = pb.ScaleOffsetRecord


def _set_attrs(node, **int_attrs):
    helper = AttrProtoHelper(node)
    for name, value in int_attrs.items():
        helper.set_attr_value(name, 'INT', value)


def _new_prune_record():
    recs = REC()
    return recs, recs.prune_record.add()


class TestPruneRecordHelperStatic(unittest.TestCase):
    def test_get_range_from_producer(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, begin=0, end=16)
        self.assertEqual(PRH.get_range(pr, 'conv1'), [0, 16])

    def test_get_range_from_consumer(self):
        _, pr = _new_prune_record()
        cons = pr.consumer.add()
        cons.name = 'conv2'
        _set_attrs(cons, begin=4, end=8)
        self.assertEqual(PRH.get_range(pr, 'conv2'), [4, 8])

    def test_get_range_not_found(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        self.assertRaises(RuntimeError, PRH.get_range, pr, 'ghost')

    def test_get_prune_group_default(self):
        _, pr = _new_prune_record()
        pr.producer.add().name = 'conv1'
        self.assertEqual(PRH.get_prune_group(pr), 1)

    def test_get_prune_group_value(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, prune_group=4)
        self.assertEqual(PRH.get_prune_group(pr), 4)

    def test_set_prune_group_new(self):
        _, pr = _new_prune_record()
        pr.producer.add().name = 'conv1'
        self.assertTrue(PRH.set_prune_group(pr, 2))
        self.assertEqual(PRH.get_prune_group(pr), 2)

    def test_set_prune_group_divisible(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, prune_group=4)
        # cur(4) % new(2) == 0 -> keep, return True
        self.assertTrue(PRH.set_prune_group(pr, 2))

    def test_set_prune_group_upgrade(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, prune_group=2)
        # new(4) % cur(2) == 0 -> upgrade to 4
        self.assertTrue(PRH.set_prune_group(pr, 4))
        self.assertEqual(PRH.get_prune_group(pr), 4)

    def test_set_prune_group_incompatible(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, prune_group=3)
        self.assertFalse(PRH.set_prune_group(pr, 2))

    def test_get_prune_axis_none(self):
        _, pr = _new_prune_record()
        pr.producer.add().name = 'conv1'
        self.assertIsNone(PRH.get_prune_axis(pr))

    def test_get_prune_axis_value(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, prune_axis=1)
        self.assertEqual(PRH.get_prune_axis(pr), 1)

    def test_get_branch_record_keys_no_branch(self):
        _, pr = _new_prune_record()
        cons = pr.consumer.add()
        cons.name = 'split1'
        keys = PRH.get_branch_record_keys(pr, 'split1')
        self.assertEqual(keys, ['passive_prune_records'])

    def test_get_branch_record_keys_with_branch(self):
        _, pr = _new_prune_record()
        cons = pr.consumer.add()
        cons.name = 'split1'
        _set_attrs(cons, branch_idx=2)
        keys = PRH.get_branch_record_keys(pr, 'split1')
        self.assertEqual(keys, ['passive_prune_records:2'])

    def test_get_branch_record_keys_branch_zero(self):
        _, pr = _new_prune_record()
        cons = pr.consumer.add()
        cons.name = 'split1'
        _set_attrs(cons, branch_idx=0)
        keys = PRH.get_branch_record_keys(pr, 'split1')
        self.assertEqual(keys, ['passive_prune_records'])

    def test_read_attr_from_proto(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        helper = AttrProtoHelper(prod)
        helper.set_attr_value('remain_channels', 'INTS', [0, 1, 2])
        helper.set_attr_value('begin', 'INT', 0)
        helper.set_attr_value('end', 'INT', 3)
        remain, begin, end = PRH.read_attr_from_proto(prod)
        self.assertEqual(list(remain), [0, 1, 2])
        self.assertEqual(begin, 0)
        self.assertEqual(end, 3)

    def test_delete_consumer_from_record(self):
        _, pr = _new_prune_record()
        c1 = pr.consumer.add()
        c1.name = 'c1'
        c2 = pr.consumer.add()
        c2.name = 'c2'
        PRH.delete_consumer_from_record(pr, 'c1')
        names = [c.name for c in pr.consumer]
        self.assertEqual(names, ['c2'])

    def test_prepare_split_info(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, begin=0)
        cons = pr.consumer.add()
        cons.name = 'conv2'
        _set_attrs(cons, begin=4)
        info = PRH.prepare_split_info([pr])
        self.assertIn('conv1', info)
        self.assertIn('active_prune_split', info['conv1'])
        self.assertIn('conv2', info)
        self.assertIn('passive_prune_split', info['conv2'])

    def test_parse_record_proto_to_dict(self):
        _, pr = _new_prune_record()
        prod = pr.producer.add()
        prod.name = 'conv1'
        ph = AttrProtoHelper(prod)
        ph.set_attr_value('remain_channels', 'INTS', [0, 1])
        ph.set_attr_value('begin', 'INT', 0)
        ph.set_attr_value('end', 'INT', 2)
        cons = pr.consumer.add()
        cons.name = 'conv2'
        ch = AttrProtoHelper(cons)
        ch.set_attr_value('remain_channels', 'INTS', [0])
        ch.set_attr_value('begin', 'INT', 0)
        ch.set_attr_value('end', 'INT', 1)
        active, passive = PRH.parse_record_proto_to_dict([pr])
        self.assertIn('conv1', active)
        self.assertIn('conv2', passive)


class TestPruneRecordHelperInstance(unittest.TestCase):
    def test_add_record(self):
        recs = REC()
        helper = PRH(recs, graph=None)
        rec = helper.add_record()
        self.assertEqual(len(recs.prune_record), 1)
        self.assertIs(rec, recs.prune_record[0])

    def test_get_record_cout(self):
        recs = REC()
        pr = recs.prune_record.add()
        prod = pr.producer.add()
        prod.name = 'conv1'
        _set_attrs(prod, begin=0, end=16)
        helper = PRH(recs, graph=None)
        self.assertEqual(helper.get_record_cout(pr), 16)

    def test_delete_redundant_attr(self):
        recs = REC()
        pr = recs.prune_record.add()
        prod = pr.producer.add()
        prod.name = 'conv1'
        ph = AttrProtoHelper(prod)
        ph.set_attr_value('begin', 'INT', 0)
        ph.set_attr_value('tmp', 'INT', 1)
        helper = PRH(recs, graph=None)
        helper.delete_redundant_attr(['tmp'])
        names = [a.name for a in prod.attr]
        self.assertNotIn('tmp', names)


if __name__ == '__main__':
    unittest.main()
