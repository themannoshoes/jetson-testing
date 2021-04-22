#ifndef _MJ_STORE_TEST_
#define _MJ_STORE_TEST_

#include <stdio.h>
#include <stdint.h>
#include "packer.h"


#define BUFFER_HEAD_ADDR 0xC0000000
#define BUFFER_END_ADDR  0xC0800000


#define FRAME_BUF_SIZE  250

#define TIMESTAMP_TYPE uint32_t
#define SIZE_OF_DATA(data) sizeof(data)

typedef enum _STORE_DATA_TYPE_ID{
    ST_ID_UINT8,
    ST_ID_UINT16,
    ST_ID_UINT32,
    ST_ID_UINT64,
    ST_ID_INT8,
    ST_ID_INT16,
    ST_ID_INT32,
    ST_ID_INT64,
    ST_ID_FLOAT,
    ST_ID_DOUBLE
}S_TYPE_ID;

typedef struct _data_information{
    uint16_t   data_name_id;
    S_TYPE_ID  data_type_id;
    uint8_t    data_size;
    int8_t  _8_;
    int16_t _16_;
    int32_t _32_;
    int64_t _64_;
    float    _f_;
    double   _db_;
    
}data_info;

typedef struct _store_structure{
    uint8_t * current_write_cursor;    //the current cursor we can write
    uint8_t * current_read_cursor;  //the reading cursor when we read out data of store_memory, it point to the current buffer we haven't read data out yet
    uint8_t * store_mem_head_p; //the head pointer, it point to the head buffer that store the first data
    uint8_t * store_mem_tail_p; //the tail pointer, it point to the tail buffer that store the last data
    uint8_t * unit_table;
 
    uint16_t data_num_in_unit;    //the num of an unit data
    uint16_t unit_data_dim;      //the dimension of an unit data
    uint8_t start_flag;
}store_type;




extern data_info temp_one_data[3 * 4];


void init_store_sturcture(store_type * store_p, uint8_t * buf_head_addr, uint8_t *buf_end_addr);
void sample_one_data_by_table(uint8_t * table_p, uint16_t data_snum, data_info * one_data);
uint16_t get_data_offset_in_the_buffer(uint8_t * table_p, uint16_t data_snum_in_table, uint8_t type_ret);
uint16_t calculate_data_unit_dim(void * table_p);
uint16_t calculate_table_dim(uint8_t * table_p);
int8_t seek_one_data_in_store_test(store_type * store_p,uint8_t * table_p, uint8_t * head_cursor_to_one_unit, uint16_t data_snum_of_unit, data_info * one_data);

void test_store_code(uint32_t timestamp);
void test_endian_transf(void);

#endif