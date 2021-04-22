#include "use_a_prtcol_to_pack_store.h"
#include "mj_store_test.h"
#include "packer.h"
#include "fops_interface.h"

extern fops_t_temp serial_fops_temp;

uart_sta_t test_store_parser(uart_t_temp *s,  void *ctx)
{
    buf_t *frm_buf = &s->_frm_buf;

    uint8_t tx_id      = 0;
    uint8_t tx_sub_id  = 0;
    uint8_t rx_id      = 0;
    uint8_t rx_sub_id  = 0;
    uint8_t frm_type   = 0;
    
    uint8_t   i = 0;
    uint8_t   j = 0;
    uint16_t  _temp_16 = 0;
    uint8_t   _temp_8  = 0;
    uint8_t   _temp1_8 = 0;
    double    _temp_DB = 0;
    uint32_t  _temp_32 = 0;
    uint8_t   ret_val;
 
    tx_id      = *(frm_buf->_base + AIRSHIP_FRAME_MASK_SEND);
    tx_sub_id  = *(frm_buf->_base + AIRSHIP_FRAME_MASK_SEND_SUB);
    frm_type   = *(frm_buf->_base + AIRSHIP_FRAME_MASK_FRM_TYPE);
    rx_id      = *(frm_buf->_base + AIRSHIP_FRAME_MASK_REV);
    rx_sub_id  = *(frm_buf->_base + AIRSHIP_FRAME_MASK_REV_SUB);
    
    store_type * store_p = (store_type *)ctx;
    
    uint8_t data_size = *(frm_buf->_base + AIRSHIP_FRAME_MASK_LEN);
    uint8_t * data_base = frm_buf->_base + AIRSHIP_FRAME_MASK_DATA;
    uint16_t store_all_unit_size = store_p->unit_data_dim + 4;
    uint16_t how_many_units_we_get = data_size / store_all_unit_size;
    uint16_t how_many_data_in_one_unit = store_p->data_num_in_unit + 1;
    
     if (tx_id == 0x99 
      && rx_id == 0x99) {
         if ((tx_sub_id == 0x01) 
          && (rx_sub_id == 0x01)){
            if (frm_type == 0xAA){
                for(uint8_t j =0; j < how_many_units_we_get; j++){
                    for(uint8_t i = 0;i < how_many_data_in_one_unit; i++){
                        seek_one_data_in_store_test(store_p , store_p->unit_table, data_base + j*store_all_unit_size, i, &temp_one_data[i+j * how_many_data_in_one_unit]);
                    }
                }
            }
          }
      }
}

void pack_most_data_in_a_frame(store_type * store_p, packer_t * gp_packer, uint32_t timestamp){  //frame_buf is the data_pointer for a airship frame
 
    uint8_t * cur_r_cursor = store_p->current_read_cursor;
    uint16_t data_unit_size = store_p->data_num_in_unit;
    uint8_t how_many_unit_can_store = FRAME_BUF_SIZE / data_unit_size;
 
    uint16_t data_size_in_a_frm = how_many_unit_can_store * data_unit_size;
    if(data_size_in_a_frm > FRAME_BUF_SIZE){
        return;
    }

    //move data into frame buf 
    memmove((void *)gp_packer->_data._base, (const void *)cur_r_cursor, data_size_in_a_frm);
    gp_packer->_timestamp     = timestamp;
    gp_packer->_tx_id         = AIRSHIP_SHIP_ID;
    gp_packer->_tx_sub_id     = AIRSHIP_FLY_CTL_BOARD_ID;
    gp_packer->_rx_id         = AIRSHIP_GCS_ID;
    gp_packer->_rx_sub_id     = AIRSHIP_FLY_CTL_BOARD_ID;
    gp_packer->_frm_type      = 0x00;   //not define yet

    packer_run(gp_packer, gp_packer->_data._base, data_size_in_a_frm, NULL);
}

void fake_uart_read(uart_t_temp  *u, uint8_t * buffer_for_read, uint16_t read_size)
{
    uint16_t bytes_read = 0;
    /* read as many bytes from serial as we can */
    uint8_t *dst = u->_raw_buf._base + u->_received;
    uint16_t bytes_to_read = u->_raw_buf._size - u->_received;

    memmove(dst, buffer_for_read,read_size);//read as more as we can
    bytes_read = read_size;

    /* update received bytes cursor */
    u->_received += bytes_read;

}

void test_parse_the_packer(uint8_t * read_buf_p, uint16_t read_size, store_type * store_p)
{
    uart_sta_t st = UART_NO_ERR;
    uart_t_temp * uart_obj_p;
    uint8_t count = 3;
    uart_obj_p = uart_create_temp(IRIDIUM2_FD, 
                                  300,     /* size of raw buffer and frame buffer */
                                  &serial_fops_temp, 
                                  airship_assemble_temp,
                                  test_store_parser);
    //read the data into buffer
    fake_uart_read(uart_obj_p, read_buf_p, read_size);
 
    while (count--) {
        st = airship_assemble_temp(uart_obj_p, NULL);

        if (st == UART_FRAME_FOUND) {
            uart_obj_p->parser(uart_obj_p, (void *)store_p);
            continue;
        } else {

            break;
        }
    }
    uart_destroy_temp(&uart_obj_p);
}

