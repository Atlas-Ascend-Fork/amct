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
import numpy as np
from ..utils.prune_record_attr_util import AttrProtoHelper
from ...utils.log import LOGGER

ATTR_BEGIN = 'begin'
ATTR_END = 'end'
PRUNE_GROUP = 'prune_group'
ACTIVE_PRUNE_SPLIT = 'active_prune_split'
PASSIVE_PRUNE_SPLIT = 'passive_prune_split'
ACTIVE_PRUNE_RECORDS = 'active_prune_records'


class PruneRecordHelper:
    """ the help class of prune record"""
    def __init__(self, records, graph):
        """
        Function: init object
        Param: node
        Return: None
        """
        self.records = records
        self.prune_records = self.records.prune_record
        self.graph = graph

    @staticmethod
    def get_range(record, name):
        """
        Function: get the range of cout
        Param: record
            name, string
        Return: a number
        """
        for producer in record.producer:
            if producer.name != name:
                continue
            attr_helper = AttrProtoHelper(producer)
            if attr_helper.has_attr(ATTR_BEGIN):
                cout_range = [attr_helper.get_attr_value(ATTR_BEGIN), attr_helper.get_attr_value(ATTR_END)]
                return cout_range
        for consumer in record.consumer:
            if consumer.name != name:
                continue
            attr_helper = AttrProtoHelper(consumer)
            if attr_helper.has_attr(ATTR_BEGIN):
                cout_range = [attr_helper.get_attr_value(ATTR_BEGIN), attr_helper.get_attr_value(ATTR_END)]
                return cout_range
        raise RuntimeError("cannot find range in record for {}".format(name))

    @staticmethod
    def get_prune_group(record):
        """
        Function: get the prune group info from one prune record
        Param: record
        Return: a number
        """
        attr_helper = AttrProtoHelper(record.producer[0])
        if attr_helper.has_attr(PRUNE_GROUP):
            prune_group = attr_helper.get_attr_value(PRUNE_GROUP)
        else:
            prune_group = 1
        return prune_group

    @staticmethod
    def set_prune_group(record, prune_group):
        """
        Function: set the prune group info to record
        Param: record
            prune_group, a number
        Return: a bool, success or not
        """
        for producer in record.producer:
            attr_helper = AttrProtoHelper(producer)
            if not attr_helper.has_attr(PRUNE_GROUP):
                attr_helper.set_attr_value(PRUNE_GROUP, 'INT', prune_group)
                return True

            cur_prune_group = attr_helper.get_attr_value(PRUNE_GROUP)
            if cur_prune_group % prune_group == 0:
                return True
            if prune_group % cur_prune_group == 0:
                attr_helper.set_attr_value(PRUNE_GROUP, 'INT', prune_group)
                return True
            return False

    @staticmethod
    def get_prune_axis(record):
        """
        Function: get the prune group info from one prune record
        Param: record
        Return: a number
        """
        attr_helper = AttrProtoHelper(record.producer[0])
        if attr_helper.has_attr("prune_axis"):
            prune_axis = attr_helper.get_attr_value("prune_axis")
        else:
            prune_axis = None
        return prune_axis

    @staticmethod
    def get_branch_record_keys(record, name):
        """
        Function: get the record's key of split node's output
        Param: record, name
        Return: key of record with branch index
        """
        branch_idx = 0
        branch_record_keys = []
        for consumer_record in record.consumer:
            if consumer_record.name != name:
                continue
            attr_helper = AttrProtoHelper(consumer_record)
            if not attr_helper.has_attr("branch_idx"):
                branch_record_keys.append('passive_prune_records')
                break
            branch_idx = attr_helper.get_attr_value("branch_idx")
            if branch_idx != 0:
                branch_record_key = 'passive_prune_records:{}'.format(branch_idx)
            else:
                branch_record_key = 'passive_prune_records'
            branch_record_keys.append(branch_record_key)
        return branch_record_keys

    @staticmethod
    def prepare_split_info(prune_records):
        """
        Function: prepare prune_split info indicating the index mapping for prune
        Param: prune_records
        Return: split_info, a dict
        """
        split_info = {}
        for prune_record in prune_records:
            for producer in prune_record.producer:
                name = producer.name
                if name not in split_info:
                    split_info[name] = {}
                if ACTIVE_PRUNE_SPLIT not in split_info.get(name):
                    split_info.get(name)[ACTIVE_PRUNE_SPLIT] = {}
                attr_helper = AttrProtoHelper(producer)
                ori_begin = attr_helper.get_attr_value(ATTR_BEGIN)
                split_info.get(name).get(ACTIVE_PRUNE_SPLIT)['ori_begin_%s' % (ori_begin)] = ori_begin
            for consumer in prune_record.consumer:
                name = consumer.name
                if name not in split_info:
                    split_info[name] = {}
                if PASSIVE_PRUNE_SPLIT not in split_info.get(name):
                    split_info.get(name)[PASSIVE_PRUNE_SPLIT] = {}
                attr_helper = AttrProtoHelper(consumer)
                ori_begin = attr_helper.get_attr_value(ATTR_BEGIN)
                split_info.get(name).get(PASSIVE_PRUNE_SPLIT)['ori_begin_%s' % (ori_begin)] = ori_begin
        return split_info

    @staticmethod
    def parse_record_proto_to_dict(prune_records):
        '''
        Function:
            transform record proto object to a dict
        input:
            prune_records: a list of proto_object of prune record
        outputs:
            active_remain_channels: a dict using layer_name as key and remain out channels as value
            passive_remain_channels: a dict using layer_name as key and remain in channels as value
        '''
        active_remain_channels_raw = {}
        passive_remain_channels_raw = {}
        for prune_record in prune_records:
            # read producer records
            for producer_record in prune_record.producer:
                if not active_remain_channels_raw.get(producer_record.name):
                    active_remain_channels_raw[producer_record.name] = {}
                remain_channels, begin, end = PruneRecordHelper.read_attr_from_proto(
                    producer_record)
                active_remain_channels_raw.get(producer_record.name)[(begin, end)] = remain_channels
            # read consumer records
            for consumer_record in prune_record.consumer:
                if not passive_remain_channels_raw.get(consumer_record.name):
                    passive_remain_channels_raw[consumer_record.name] = {}
                remain_channels, begin, end = PruneRecordHelper.read_attr_from_proto(
                    consumer_record)
                passive_remain_channels_raw.get(consumer_record.name)[(begin, end)] = remain_channels

        # remain channels may be splitted by begin and end index
        # here the splitted records need to be merged
        active_remain_channels = _merge_channels_by_begin_end(active_remain_channels_raw)
        passive_remain_channels = _merge_channels_by_begin_end(passive_remain_channels_raw)
        return active_remain_channels, passive_remain_channels

    @staticmethod
    def read_attr_from_proto(record):
        '''
        Function:
            read remain_channels, begin, and end attr from prune record
        inputs:
            record: record proto object
        '''
        attr_helper = AttrProtoHelper(record)
        remain_channels = attr_helper.get_attr_value('remain_channels')
        begin = attr_helper.get_attr_value(ATTR_BEGIN)
        end = attr_helper.get_attr_value(ATTR_END)
        return remain_channels, begin, end

    @staticmethod
    def delete_consumer_from_record(prune_record, name):
        """
        Function: delete consumer from prune_record
        param:prune_record: prune_record
        param:name: string, consumer's name to delete
        """
        del_consumer = None
        for consumer in prune_record.consumer:
            if consumer.name == name:
                del_consumer = consumer
                break
        prune_record.consumer.remove(del_consumer)

    def get_record_cout(self, record):
        """
        Function: get the prune length from one record
        Param: record
        Return: a number
        """
        cout_range = self.get_range(record, record.producer[0].name)
        cout_len = cout_range[1] - cout_range[0]
        return cout_len

    def add_record(self):
        """
        Function: add one prune record
        Param: None
        Return: None
        """
        record = self.prune_records.add()
        return record

    def delete_record_list(self, del_record_list):
        '''
        Function: delete the list of del_record from record
        Param: del_record_list, the records to delete
        Return: None
        '''
        del_record_dict = {id(record): record for record in del_record_list}
        for del_record in del_record_dict.values():
            self.delete_record(del_record)

    def delete_record(self, del_record):
        """
        Function: delete the del_record from record
        Param: del_record, the record to delete
        Return: None
        """
        # delete prune attr for related nodes
        producer_names = {producer.name for producer in del_record.producer}
        consumer_names = {consumer.name for consumer in del_record.consumer}

        for producer_name in producer_names:
            producer_node = self.graph.get_node_by_name(producer_name)
            records = producer_node.get_attr(ACTIVE_PRUNE_RECORDS)
            if del_record not in records:
                raise RuntimeError(
                    'the del_record is not in active_prune_records of producer {}'.format(producer_name))
            records.remove(del_record)
            if not records:
                producer_node.delete_attr(ACTIVE_PRUNE_RECORDS)

        for consumer_name in consumer_names:
            consumer_node = self.graph.get_node_by_name(consumer_name)
            branch_record_keys = self.get_branch_record_keys(del_record, consumer_name)
            for branch_record_key in branch_record_keys:
                records = consumer_node.get_attr(branch_record_key)
                if del_record not in records:
                    raise RuntimeError(
                        'the del_record is not in passive_prune_records of consumer {}'.format(consumer_name))
                records.remove(del_record)
                if not records:
                    consumer_node.delete_attr(branch_record_key)

        self.prune_records.remove(del_record)

    def merge_record(self, record1, record2):
        """
        Function: merge record1 and record2 to one new record
        Param: record1, the record to merge
            record1, another record to merge
        Return: new merged record
        """
        if record1 is record2:
            # no need to merge
            return record1
        producer_names = {producer.name for producer in record2.producer}
        consumer_names = {consumer.name for consumer in record2.consumer}
        record1.producer.extend(record2.producer)
        record1.consumer.extend(record2.consumer)
        for producer_name in producer_names:
            producer_node = self.graph.get_node_by_name(producer_name)
            producer_node.get_attr(ACTIVE_PRUNE_RECORDS).remove(record2)
            producer_node.get_attr(ACTIVE_PRUNE_RECORDS).append(record1)
        for consumer_name in consumer_names:
            consumer_node = self.graph.get_node_by_name(consumer_name)
            branch_record_keys = self.get_branch_record_keys(record2, consumer_name)
            for branch_record_key in branch_record_keys:
                consumer_node.get_attr(branch_record_key).remove(record2)
                consumer_node.get_attr(branch_record_key).append(record1)
        # union prune_group
        new_prune_group = 0
        for producer in record1.producer:
            attr_helper = AttrProtoHelper(producer)
            if attr_helper.has_attr(PRUNE_GROUP):
                prune_group = attr_helper.get_attr_value(PRUNE_GROUP)
                new_prune_group = max(prune_group, new_prune_group)
        if new_prune_group > 0:
            for producer in record1.producer:
                attr_helper = AttrProtoHelper(producer)
                attr_helper.set_attr_value(PRUNE_GROUP, 'INT', new_prune_group)

        # remove record
        self.prune_records.remove(record2)

        return record1

    def split_record(self, record, split_info):
        """
        Function: split the prune record according to split_info
        Param: record, a record to split
            split_info, a list
        Return: a list containing several records
        """
        # add several records and del one record
        self.prune_records.remove(record)
        records_len = len(self.prune_records)
        split_grp = len(split_info)
        for _ in range(split_grp):
            self.prune_records.add()
        # copy and modify producer info
        self._split_record_to_parts(record, split_info, records_len)

        # modify record info for producer
        producer_names = [producer.name for producer in record.producer]
        for producer_name in producer_names:
            producer_node = self.graph.get_node_by_name(producer_name)
            old_records = producer_node.get_attr(ACTIVE_PRUNE_RECORDS)
            pos = old_records.index(record)
            old_records.remove(record)
            for new_record in self.prune_records[records_len:]:
                old_records.insert(pos, new_record)
                pos += 1
        # modify record info for consumer
        consumer_names = [consumer.name for consumer in record.consumer]
        for consumer_name in consumer_names:
            consumer_node = self.graph.get_node_by_name(consumer_name)
            old_records = consumer_node.get_attr('passive_prune_records')
            pos = old_records.index(record)
            old_records.remove(record)
            for new_record in self.prune_records[records_len:]:
                old_records.insert(pos, new_record)
                pos += 1
        return self.prune_records[records_len:]

    def delete_redundant_attr(self, del_attr_names):
        """
        Function: delete redundant content in prune_record
        Param: del_attr_names, a list
        Return: None
        """
        for prune_record in self.prune_records:
            # delete some attr in producer
            for producer in prune_record.producer:
                _delete_attr(producer, del_attr_names)
            # delete some attr in consumer
            for consumer in prune_record.consumer:
                _delete_attr(consumer, del_attr_names)

    def _split_record_to_parts(self, record, split_info, records_len):
        """
        Function: split record to several parts
        Params: record to split
            split_info, a list
            records_len: a number
        Return: None
        """
        # copy and modify producer info
        for producer in record.producer:
            attr_helper = AttrProtoHelper(producer)
            if attr_helper.has_attr(ATTR_BEGIN):
                begin = attr_helper.get_attr_value(ATTR_BEGIN)
            else:
                begin = 0
            output_range = [begin, begin]
            for idx, split_size in enumerate(split_info):
                new_record = self.prune_records[records_len + idx]
                new_producer = new_record.producer.add()
                new_producer.CopyFrom(producer)
                attr_helper = AttrProtoHelper(new_producer)
                output_range = [output_range[1], output_range[1] + split_size]
                attr_helper.set_attr_value(ATTR_BEGIN, 'INT', output_range[0])
                attr_helper.set_attr_value(ATTR_END, 'INT', output_range[1])
        # copy and modify consumer info
        for consumer in record.consumer:
            attr_helper = AttrProtoHelper(consumer)
            if attr_helper.has_attr(ATTR_BEGIN):
                begin = attr_helper.get_attr_value(ATTR_BEGIN)
            else:
                begin = 0
            output_range = [begin, begin]
            for idx, split_size in enumerate(split_info):
                new_record = self.prune_records[records_len + idx]
                new_consumer = new_record.consumer.add()
                new_consumer.CopyFrom(consumer)
                attr_helper = AttrProtoHelper(new_consumer)
                output_range = [output_range[1], output_range[1] + split_size]
                attr_helper.set_attr_value(ATTR_BEGIN, 'INT', output_range[0])
                attr_helper.set_attr_value(ATTR_END, 'INT', output_range[1])


def _merge_channels_by_begin_end(remain_channels_raw):
    '''Function: concatnate all remain channels for each layer'''
    remain_channels = {}
    for layer, chanels in remain_channels_raw.items():
        # sort dict items by begin value
        remain_channel = []
        sorted_remain_channels = sorted(chanels.items(), key=lambda x: x[0][0])

        # concatenate all remain channels
        for sorted_remain_channel in sorted_remain_channels:
            # remain_channels shifted by begin
            shifted_remain_channels = sorted_remain_channel[0][0] + np.asarray(sorted_remain_channel[1])
            remain_channel.extend(shifted_remain_channels.tolist())
        remain_channels[layer] = remain_channel

    return remain_channels


def _delete_attr(record, del_attr_names):
    '''Function: delete target attr in record'''
    del_attrs = []
    for attr in record.attr:
        if attr.name in del_attr_names:
            del_attrs.append(attr)
    for attr in del_attrs:
        record.attr.remove(attr)
