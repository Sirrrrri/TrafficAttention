#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
MFR数据处理模块
结合YaTC的MFR流量表达方式和ET-BERT的数据处理方法
"""

from .data_process import (
    read_MFR_bytes,
    read_MFR_sequence,
    read_MFR_token_sequence,
    process_pcap_dataset,
    convert_json_to_npy,
    generate_vocab_from_mfr,
    save_vocab,
    makedir
)

__version__ = '1.0.0'
__author__ = 'ET-BERT + YaTC Integration'

__all__ = [
    'read_MFR_bytes',
    'read_MFR_sequence',
    'read_MFR_token_sequence',
    'process_pcap_dataset',
    'convert_json_to_npy',
    'generate_vocab_from_mfr',
    'save_vocab',
    'makedir'
]

