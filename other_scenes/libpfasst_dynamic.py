"""
This file contains a Manim Scene that visualizes
the solving of the 2d heat equation parallel in time using LibPFASST
It also shows how the application is able to grow and shrink dynamically

Due to bugs in Manim, it only work using the opengl renderer
It was developed and tested with manim==0.17.3 (The community edition)

For rapid development, this command worked best for me (from the parent dir):

manim render --disable_caching -p -qm ./other_scenes/libpfasst_dynamic.py --renderer opengl
"""
from manim import *
from manim.opengl import *

import matplotlib.pyplot as plt
import numpy as np





class HeatEqu(ThreeDScene):
    def construct(self):
        ## CONFIG
        phi_deg = 70
        theta_deg = 135
        self.set_camera_orientation(phi=phi_deg * DEGREES, theta=theta_deg * DEGREES)

        self.t0 = 0
        self.num_timesteps = 3
        self.num_space = 4 # must be square number
        self.dt = 0.005

        self.cur_step = 0

        self.sidelen_c = np.pi

        self.num_space_side = int(np.sqrt(self.num_space))

        self.plot_resolution = 0.1

        self.z_min = -20*self.dt
        self.z_max = 20*self.dt

        self.cur_axes_scale = 0

        # process mobjects on the axes
        self.processes = []

        # pv = process view (bottom left)
        self.pv_procs = []
        self.pv_procs_group = VGroup()
        self.pv_width = 2
        self.pv_new_pset_buff = 0.3

        self.panes = []


        self.split_colors = [RED, BLUE, GREEN, PURPLE, ORANGE, PINK, YELLOW]
        if self.num_space > 1:
            # create cycle
            self.split_colors *= int(self.num_space / len(self.split_colors)) + 1
        self.split_colors = self.split_colors[:self.num_space]

        self.space_pset_color = "#F5F5DC" # beige



        ## STEP: Setup Axes
        self.update_text(f"<b>2D Heat Equation</b>\n\n<i>parallel and\nadaptive in time</i>")
        self.setup_axes()
        self.play(FadeIn(self.cur_text), FadeIn(self.axes), FadeIn(self.axes_labels), *[FadeIn(v) for v in self.grids.values()], FadeIn(self.time_arrow))
        self.play(FadeOut(self.cur_text))




        ## STEP: show mpi://WORLD
        self.update_text(f"Initial pset\n<span foreground='yellow'>mpi://WORLD</span> with\n"
                         f"<span foreground='green'>{self.num_space * self.num_timesteps} processes</span>")

        # create new pset in process view (bottom left)
        self.pv_procs_group, self.pv_procs, world_box = self.pv_new_pset(self.num_timesteps)
        self.play(FadeIn(self.cur_text), FadeIn(self.pv_procs_group))

        self.play(FadeOut(self.cur_text))




        ## STEP: show processes and communicators on axes
        # put processes onto axis
        self.update_text("Each process is assigned\na space slice\nand a timestep")

        self.processes = self.create_axes_processes(self.num_timesteps, self.t0 + self.cur_step*self.dt)
        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in self.processes])

        # remove world_box
        self.pv_procs_group.remove(world_box)
        self.play(FadeOut(self.cur_text), FadeOut(world_box))



        ### STEP: split up mpi://WORLD
        space_boxes, space_arrows, time_boxes, time_arrows = self.pv_split_set_grid(self.pv_procs_group)
        self.update_text("Split up <span foreground='yellow'>mpi://WORLD</span>\ninto grid")
        self.add(*time_boxes)
        self.add(*space_boxes)
        self.bring_to_front(self.pv_procs_group)
        self.play(FadeIn(self.cur_text))
        self.play(FadeOut(self.cur_text))


        # # TODO: for some reason, arrows are not shown
        # self.update_text("Split into space psets")
        # self.add(*space_boxes)
        # self.add(*space_arrows)
        # self.bring_to_back(*space_boxes)
        # self.play(FadeIn(self.cur_text))
        # self.remove(*space_arrows)
        # self.play(FadeOut(self.cur_text))

        # self.update_text("Split into time psets")
        # self.add(*time_boxes)
        # self.add(*time_arrows)
        # self.bring_to_back(*time_boxes)
        # self.play(FadeIn(self.cur_text))
        # self.remove(*time_arrows)
        # self.play(FadeOut(self.cur_text))





        ## STEP: highlight pfasst psets
        space_psets = self.create_space_pset_boxes(self.t0 + self.cur_step*self.dt, self.num_timesteps)
        time_psets = self.create_pfasst_pset_boxes(self.t0 + self.cur_step*self.dt, self.num_timesteps)

        self.update_text(f"Space communicators\nfrom space psets")
        # TODO: for some reason, FadeIn does not work here
        self.add(*space_psets)
        # self.play(*[FadeIn(p) for p in space_psets], FadeIn(self.cur_text))
        self.play(FadeIn(self.cur_text), *[FadeOut(p) for p in time_boxes])
        self.wait(0.5)
        self.add(*time_boxes)
        self.bring_to_front(*space_boxes)
        self.bring_to_front(self.pv_procs_group)
        self.play(*[FadeOut(p) for p in space_psets], FadeOut(self.cur_text), *[FadeIn(p) for p in time_boxes])

        self.update_text(f"Time communicators\nfrom time psets")
        self.play(*[FadeIn(p) for p in time_psets], FadeIn(self.cur_text), *[FadeOut(p) for p in space_boxes])
        self.play(*[FadeOut(p) for p in time_psets], FadeOut(self.cur_text))

        # add them back and show in next animation
        self.add(*space_boxes)
        self.bring_to_front(self.pv_procs_group)



        ## STEP: show initial condition
        self.update_text("Setup initial condition")
        self.panes = self.create_initial_condition_planes(self.num_timesteps, self.t0 + self.cur_step*self.dt)

        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in self.panes])
        self.play(FadeOut(self.cur_text))


        ## STEP: solve pfasst block
        self.update_text("Solve PFASST block")
        animations = self.solve_animations()
        self.play(*animations, FadeIn(self.cur_text))
        self.play(FadeOut(self.cur_text))



        ## STEP: spread solution
        self.update_text("Bcast solution for next\ntime block")
        animations1, animations2 = self.spread_solution_animations()
        self.play(*animations1, FadeIn(self.cur_text))
        self.play(*animations2, FadeOut(self.cur_text))



        # STEP: move everything down by number of timesteps
        self.move_next_block()





        ## STEP: Addition of new processes
        # "Zoom out"
        #
        ## Add pset process sets back
        vg = VGroup()
        vg.add(self.axes)
        vg.add(*self.grids.values())
        vg.add(*self.processes)
        # g.add(*panes)
        animations = []
        animations.append(vg.animate.scale(0.6).shift(OUT*2))
        self.play(*animations)

        # update c2p
        self.sidelen_p = (self.axes.c2p(self.sidelen_c,0,0)-self.axes.c2p(0,0,0))[0]
        self.dt_p = (self.axes.c2p(0,0,self.dt) - self.axes.c2p(0,0,0))[2]

        for proc in self.processes:
            vg.remove(proc)
        for proc in self.processes:
            vg.add(proc)




        self.update_text("Case 1:\nAddition of new processes")

        self.play(FadeIn(self.cur_text))
        self.play(FadeOut(self.cur_text))


        ## STEP: show new pset
        num_new_timesteps = 2

        animations = []
        self.update_text("Runtime:\nnew set <span foreground='yellow'>mpi://add_0</span>")
        # create new pset in process view (bottom left)
        new_pv_procs_group, new_pv_procs, new_world_box = self.pv_new_pset(num_new_timesteps)
        animations.append(FadeIn(new_pv_procs_group))

        # Also add new circles on 3d axes
        new_processes = self.create_axes_processes(num_new_timesteps, self.t0 + (self.cur_step+self.num_timesteps)*self.dt)
        animations += [FadeIn(p) for p in new_processes]

        self.play(*animations, FadeIn(self.cur_text))

        # remove world_box
        new_pv_procs_group.remove(new_world_box)
        self.play(FadeOut(self.cur_text), FadeOut(new_world_box))


        ### STEP: split up new processes
        new_space_boxes, _, new_time_boxes, _ = self.pv_split_set_grid(new_pv_procs_group)
        new_time_psets = self.create_pfasst_pset_boxes(self.t0 + (self.cur_step + self.num_timesteps)*self.dt, num_new_timesteps)

        self.update_text("Split up <span foreground='yellow'>mpi://add_0</span>\ninto grid")
        self.add(*new_time_boxes, *new_space_boxes)
        self.bring_to_front(new_pv_procs_group)

        # also show old time psets
        old_time_psets = self.create_pfasst_pset_boxes(self.t0 + self.cur_step*self.dt, self.num_timesteps)
        animations = []
        for box in old_time_psets:
            # make small gap with the boxes
            height = (box.get_zenith() - box.get_nadir())[2]
            box.stretch(0.90, 2)
            new_height = (box.get_zenith() - box.get_nadir())[2]
            box.shift(IN*(height-new_height)/2)
            animations.append(FadeIn(box))

        animations += [FadeIn(box) for box in new_time_psets]
        self.play(FadeIn(self.cur_text), *animations)
        self.play(FadeOut(self.cur_text))


        ### STEP: merge time psets
        # create boxes for new splits
        self.update_text(f"{self.num_space} union operations:\n\nMerge new time psets\nwith old time psets")
        joined_time_psets = self.create_pfasst_pset_boxes(self.t0+self.cur_step*self.dt, self.num_timesteps+num_new_timesteps)

        # fade out space related stuff and fade in new time psets on 3d axes
        animations = []
        animations += [FadeOut(space_box) for space_box in space_boxes]
        animations += [FadeOut(new_space_box) for new_space_box in new_space_boxes]
        self.play(*animations, FadeIn(self.cur_text))

        # merge time psets

        # merge in bottom left process view
        # remove new pset boxes
        animations += [FadeOut(new_time_box) for new_time_box in new_time_boxes]


        # update some variables
        # ungroup old vgroups
        self.pv_procs_group.remove(*self.pv_procs)
        new_pv_procs_group.remove(*new_pv_procs)

        new_pv_p = []
        new_p = []
        for i in range(self.num_space):
            new_pv_p += [self.pv_procs[i*self.num_timesteps + j] for j in range(self.num_timesteps)]
            new_pv_p += [new_pv_procs[i*num_new_timesteps + j] for j in range(num_new_timesteps)]
            new_p += [self.processes[i*self.num_timesteps + j] for j in range(self.num_timesteps)]
            new_p += [new_processes[i*num_new_timesteps + j] for j in range(num_new_timesteps)]
        self.pv_procs = new_pv_p
        self.pv_procs_group = VGroup(*new_pv_p)
        self.processes = new_p
        self.num_timesteps = self.num_timesteps + num_new_timesteps


        # rearrange into grid
        # animations.append(self.pv_procs_group.animate.arrange_in_grid(cols=self.num_space, flow_order="ul"))
        self.pv_procs_group.arrange_in_grid(cols=self.num_space, flow_order="ul")
        animations.append(self.pv_procs_group.animate.to_edge(DOWN))

        self.play(*animations)


        # create new boundary boxes
        new_space_boxes, _, new_time_boxes, _ = self.pv_split_set_grid(self.pv_procs_group)

        # transform old time boxes into new time boxes
        animations = []
        for box,new_time_box in zip(time_boxes, new_time_boxes):
            animations.append(box.animate.become(new_time_box))

        # merge on 3d axes
        for (pset1,pset2,joined) in zip(new_time_psets, old_time_psets, joined_time_psets):
            animations += [FadeOut(pset1), FadeOut(pset2), FadeIn(joined)]

        self.play(*animations)

        animations = [FadeIn(new_space_box) for new_space_box in new_space_boxes]
        self.play(*animations, FadeOut(self.cur_text))

        # update more variables
        time_boxes = new_time_boxes
        space_boxes = new_space_boxes

        # remove time psets and add initial conditions
        self.panes = self.create_initial_condition_planes(self.num_timesteps, self.t0 + self.cur_step*self.dt)
        animations = []
        animations += [FadeOut(pset) for pset in joined_time_psets]
        animations += [FadeIn(p) for p in self.panes]
        self.play(*animations)


        ## STEP: solve pfasst block
        self.update_text(f"Solve PFASST block\nwith <span foreground='green'>{self.num_timesteps}</span> parallel time steps")
        animations = self.solve_animations()
        self.play(*animations, FadeIn(self.cur_text))
        self.play(FadeOut(self.cur_text))


        ## STEP: spread solution
        self.update_text("Bcast solution for next\ntime block")
        animations1, animations2 = self.spread_solution_animations()
        self.play(*animations1, FadeIn(self.cur_text))
        self.play(*animations2, FadeOut(self.cur_text))



        # STEP: move everything down by number of timesteps
        self.move_next_block()



        self.update_text("Case 2:\nRemoval of existing processes")

        self.play(FadeIn(self.cur_text))
        self.play(FadeOut(self.cur_text))

        self.wait(2)








    def setup_axes(self):
        self.axes = ThreeDAxes(
                x_range=[-0.1*self.sidelen_c,1.5*self.sidelen_c,1],
                x_length=8,
                y_range=[-0.1*self.sidelen_c,1.5*self.sidelen_c,1],
                y_length=8,
                z_range=[self.z_min,self.z_max,self.dt/5],
                z_length=3*(self.z_max-self.z_min)/self.dt,
                )


        self.axes.scale(0.65)
        self.cur_axes_scale = 0.65
        # self.axes.shift(2*OUT+4*UP)
        self.axes.shift(3.2*IN)

        self.sidelen_p = (self.axes.c2p(self.sidelen_c,0,0)-self.axes.c2p(0,0,0))[0]
        self.dt_p = (self.axes.c2p(0,0,self.dt) - self.axes.c2p(0,0,0))[2]

        self.axes_labels = self.axes.get_axis_labels(
                "x", "y", "t"
                )

        self.time_arrow = Text("Time →")

        #TODO: is nicer with fixed_frame
        self.time_arrow.rotate(90*DEGREES, axis=RIGHT)
        self.time_arrow.rotate(270*DEGREES, axis=UP)
        self.time_arrow.rotate(180*DEGREES, axis=OUT)
        self.time_arrow.scale(0.4)
        self.time_arrow.move_to(self.axes.c2p(0,0,3.2*self.dt) + 0.5*RIGHT)

        # # remove x and y axes
        # self.axes.remove(self.axes[0], self.axes[1])

        # setup grid + label along each timestep
        self.grids = {}
        for i in range(int(self.z_max/self.dt)):
            t = self.t0 + i*self.dt
            grid = VGroup(
                *[Square(side_length=self.sidelen_p/self.num_space_side) for _ in range(self.num_space)]
                   ).arrange_in_grid(rows=self.num_space_side,cols=self.num_space_side,buff=0)
            grid.move_to(self.axes.c2p(self.sidelen_c/2,self.sidelen_c/2,t))
            # add time label to it
            t_label = MathTex(f"t_0" if i == 0 else f"t_0 + {i}dt")
            t_label.rotate(90*DEGREES, axis=OUT)
            t_label.rotate(90*DEGREES, axis=UP)
            t_label.move_to(self.axes.c2p(-0.5, self.sidelen_c/2,t))
            self.bring_to_back(t_label)
            vg = VGroup(grid, t_label)
            self.grids[t] = vg


    def pv_new_pset(self, num_new_timesteps):
        """
        create new pset for the process view
        """
        # create processes as circles
        circles = []
        for i in range(self.num_space * num_new_timesteps):
            c = self.create_process_circle(len(self.pv_procs) + i)
            circles.append(c)

        procs_group = VGroup(*circles).arrange_in_grid(cols=self.num_space, flow_order='ul')
        procs_group.scale(0.9 * self.pv_width / procs_group.width)
        procs_group.arrange_in_grid(cols=self.num_space, flow_order='ul')

        # put box around them
        world_box = SurroundingRectangle(procs_group)
        procs_group.add(world_box)

        self.add_fixed_in_frame_mobjects(procs_group)
        self.remove(procs_group)

        if len(self.pv_procs_group) > 0:
            procs_group.next_to(self.pv_procs_group, direction=UP, buff=self.pv_new_pset_buff)
        else:
            procs_group.to_edge(DOWN+LEFT)

        return procs_group, circles, world_box


    def create_axes_processes(self, num_new_timesteps, base_time):
        circles = []
        for i in range(self.num_space * num_new_timesteps):
            c = self.create_process_circle(len(self.processes) + i)
            c.rotate(90*DEGREES, axis=OUT)
            c.scale(0.3*self.sidelen_p / c.width)

            pos = self.get_proc_pos_c(i, base_time, num_new_timesteps)
            c.move_to(self.axes.c2p(*pos))
            circles.append(c)

        return circles


    def create_initial_condition_planes(self, num_timesteps, base_time):
        panes = [self.get_value_pane(base_time) for _ in range(num_timesteps)]

        # fix position for some of the things
        for i,pane in enumerate(panes):
            pos = self.get_valuepane_pos(base_time+i*self.dt)
            pane.move_to(self.axes.c2p(*pos))
            self.bring_to_back(pane)

        return panes


    def solve_animations(self):
        ## STEP: solve pfasst block
        panes = [self.get_value_pane(self.t0+(self.cur_step + i)*(self.dt)) for i in range(1,self.num_timesteps+1)]

        animations = []
        # for p in self.panes[1:]:
        #     animations.append(FadeOut(p))
        self.remove(*self.panes[1:])

        # don't hide grid
        for p in panes:
            self.bring_to_back(p)
            animations.append(FadeIn(p))


        self.panes = [self.panes[0]] + panes

        return animations


    def spread_solution_animations(self):
        ## STEP: spread solution
        # remove panes
        animations1 = [FadeOut(p) for p in self.panes[:-1]]

        # copy last pane to rest
        for i in range(len(self.panes)-1):
            self.panes[i] = self.panes[-1].copy()
            self.bring_to_back(self.panes[i])

        # move each pane to its position
        animations2 = []
        for i,p in enumerate(self.panes[:-1]):
            pos = self.get_valuepane_pos(self.t0+(self.cur_step + i)*self.dt)
            animations2.append(p.animate.move_to(self.axes.c2p(*pos)))

        # remove last pane
        self.remove(self.panes[-1])
        self.panes = self.panes[:-1]

        return animations1, animations2


    def move_next_block(self):
        self.update_text("Move to next block")
        animations = []
        for g in self.grids.values():
            animations.append(g.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes_labels.animate.shift(IN*self.dt_p*self.num_timesteps))

        # remove panes for the zoom out
        # remove panes as they can't be scaled in opengl
        animations += [FadeOut(p) for p in self.panes]

        self.play(*animations, FadeIn(self.cur_text), run_time=1)
        self.play(FadeOut(self.cur_text))
        self.cur_step += self.num_timesteps


    def pv_split_set_grid(self, procs):
        num_timesteps = len(procs)//self.num_space
        # create boxes around each space set
        space_sets = [VGroup(*procs[i::num_timesteps]) for i in range(num_timesteps)]
        space_boxes = []
        space_arrows = []
        for i,s in enumerate(space_sets):
            box = SurroundingRectangle(s, color=self.space_pset_color, fill_color=self.space_pset_color, fill_opacity=0.2)
            self.add_fixed_in_frame_mobjects(box)
            self.remove(box)
            space_boxes.append(box)

            rpos = box.get_center()+(box.width/2+0.1)*RIGHT
            sa = Arrow(start=rpos, end=rpos+0.5*RIGHT, color=WHITE)
            self.add_fixed_in_frame_mobjects(sa)
            self.remove(sa)
            space_arrows.append(sa)


        # create boxes around each time set
        time_sets = [VGroup(*procs[i*num_timesteps:(i+1)*num_timesteps][::-1]) for i in range(self.num_space)]
        time_boxes = []
        time_arrows = []
        for i,s in enumerate(time_sets):
            box = SurroundingRectangle(s, color=self.split_colors[i], fill_color=self.split_colors[i], fill_opacity=0.7)
            self.add_fixed_in_frame_mobjects(box)
            self.remove(box)
            time_boxes.append(box)

            upos = box.get_center()+(box.height/2+0.1)*UP
            ta = Arrow(start=upos, end=upos+0.5*UP, color=WHITE)
            self.add_fixed_in_frame_mobjects(ta)
            self.remove(ta)
            time_arrows.append(ta)

        return space_boxes, space_arrows, time_boxes, time_arrows

    def update_text(self, new_text, **kwargs):
        new_text = MarkupText(new_text, **kwargs).scale(0.65)
        new_text.to_edge(UP+LEFT)
        # need to add and remove to be able to have it in a fixed in frame with effect
        self.add_fixed_in_frame_mobjects(new_text)
        self.remove(new_text)
        self.cur_text = new_text

    def get_valuepane_pos(self, timestep):
        return np.array([self.sidelen_c/2,self.sidelen_c/2,timestep])


    def create_process_circle(self, i):
        #c = Circle(radius=1, fill_color=GREEN, fill_opacity=0.7)
        c = Circle(radius=1, fill_color=GREEN, fill_opacity=0.9)
        t = Tex(f"{i}")
        t.scale(2 * c.radius)
        t.move_to(c)
        v = VGroup(c, t)
        return v

    def get_value_pane(self, timestep):
        im = OpenGLImageMobject(get_heat_image(0, self.sidelen_c, 0, self.sidelen_c, self.plot_resolution, timestep))
        im.scale(self.sidelen_p/im.width)
        im.move_to(self.axes.c2p(*self.get_valuepane_pos(timestep)))

        return im

    def get_proc_pos_c(self, p_num, base_t, num_timesteps_height):
        """
        get proc position in axis coordinates
        """
        # get time step
        t = base_t + self.dt*(p_num%num_timesteps_height)
        i = p_num//num_timesteps_height

        z_pos = t
        x_pos = (0.5 + i%self.num_space_side)/self.num_space_side * self.sidelen_c
        y_pos = (0.5 + i//self.num_space_side)/self.num_space_side * self.sidelen_c
        return np.array([x_pos, y_pos, z_pos])

    def create_pfasst_pset_boxes(self, base_t, num_timesteps_height):
        psets = []
        for i in range(self.num_space):
            # create prism that spans each split
            prism = Prism(dimensions=[1,1,1], fill_color=self.split_colors[i], fill_opacity=0.2, stroke_width=0)
            for dim, factor in zip(range(3), [self.sidelen_p/self.num_space_side, self.sidelen_p/self.num_space_side, self.dt_p*num_timesteps_height]):
                prism.stretch(factor/2, dim)
            psets.append(prism)

            # cylinder = Cylinder(height=dt_p*(num_timesteps-1), radius=0.2, fill_color=split_colors[i], fill_opacity=0.2, stroke_width=0)
            pos = self.get_proc_pos_c(i*num_timesteps_height, base_t, num_timesteps_height)
            # fix z position
            pos[2] = base_t + self.dt*((num_timesteps_height)/2)
            prism.move_to(self.axes.c2p(*pos))
            # print(radius)
            # print(height)
            # radius = self.processes[0].width/2
            # height = self.dt_p*((num_timesteps_height)/2)
            # cylinder = Cylinder(radius=radius, height=height, fill_color=self.split_colors[i], fill_opacity=0.2, show_ends=False)
            # cylinder = Cylinder(radius=radius, height=height, show_ends=False)
            # cylinder.move_to(self.axes.c2p(*pos))
            # psets.append(cylinder)
        return psets


    def create_space_pset_boxes(self, base_t, num_timesteps_height):
        psets = []
        for i in range(num_timesteps_height):
            # create prism that spans each split
            prism = Prism(dimensions=[1,1,1], fill_color=self.space_pset_color, fill_opacity=0.2, stroke_width=0)
            factors = [self.sidelen_p, self.sidelen_p, self.dt_p*2/3]
            for dim, factor in zip(range(3), factors):
                prism.stretch(factor/2, dim)
            psets.append(prism)

            # cylinder = Cylinder(height=dt_p*(num_timesteps-1), radius=0.2, fill_color=split_colors[i], fill_opacity=0.2, stroke_width=0)
            pos = self.get_valuepane_pos(base_t + self.dt*i)
            pos = self.axes.c2p(*pos)
            # fix z position
            pos[2] += factors[-1]/2
            prism.move_to(pos)
            psets.append(prism)
        return psets


def get_heat_image(x_min, x_max, y_min, y_max, resolution, t):
    num_x = int((x_max-x_min)/resolution) + 1
    num_y = int((y_max-y_min)/resolution) + 1
    im = np.zeros((num_y,num_x))
    for j in range(num_y):
        for i in range(num_x):
            x = x_min + i*resolution
            y = y_min + j*resolution
            im[j,i]  =   (np.sin(x) * np.exp(-1.0 * np.pi**2.0 * t)) \
                       * (np.sin(y) * np.exp(-1.0 * np.pi**2.0 * t))

    # no need to normalize as max value is exactly 1
    cmap = plt.get_cmap('viridis')
    rgba_im = cmap(im)
    rgb_im = np.delete(rgba_im, 3, 2)
    
    # convert to uint8
    rgb_im = np.uint8(rgb_im * 256)
    return rgb_im
