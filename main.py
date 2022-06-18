import Software_Practical

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rgb_img = True
    img_height = 100
    img_width = 100
    total_img = 5
    line_width = 1
    line_color = 'blue'
    voronoi_cell_no = 8

    if rgb_img:
        tensor_3 = Software_Practical.vor_rgb_img_creator(total_img=total_img, rand_point_no=voronoi_cell_no,
                                               img_height=img_height,
                                               img_width=img_width, line_width=line_width, line_color=line_color)
        tensor = Software_Practical.tensor_out_colored(total_img=total_img, img_height=img_height,
                                                       img_width=img_width)

        tensor_2 = Software_Practical.tensor_out_2(total_img=total_img, img_height=img_height, img_width=img_width,
                                                   line_width=line_width)
    else:
        Software_Practical.vor_bw_img_creator(total_img=total_img, rand_point_no=voronoi_cell_no, img_height=img_height,
                                              img_width=img_width, line_width=line_width, line_color=line_color)
        tensor = Software_Practical.tensor_out_bw(total_img=total_img, img_height=img_height, img_width=img_width)
