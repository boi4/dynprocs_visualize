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

        self.processes = []
        self.cur_axes_scale = 0
        panes = []


        self.split_colors = [RED, BLUE, GREEN, PURPLE, ORANGE, PINK, YELLOW]
        if self.num_space > 1:
            # create cycle
            self.split_colors *= int(self.num_space / len(self.split_colors)) + 1
        self.split_colors = self.split_colors[:self.num_space]

        self.space_pset_color = "#F5F5DC"

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
        self.axes.shift(3.2*IN)

        self.sidelen_p = (self.axes.c2p(self.sidelen_c,0,0)-self.axes.c2p(0,0,0))[0]
        self.dt_p = (self.axes.c2p(0,0,self.dt) - self.axes.c2p(0,0,0))[2]

        self.axes_labels = self.axes.get_axis_labels(
                "x", "y", "t"
                )

        # self.add_fixed_orientation_mobjects(axes)

        # self.add_fixed_orientation_mobjects(labels)
        self.add(self.axes, self.axes_labels)

        t = Text("Time →")

        t.rotate(90*DEGREES, axis=RIGHT)
        t.rotate(270*DEGREES, axis=UP)
        t.rotate(180*DEGREES, axis=OUT)
        t.scale(0.4)
        t.move_to(self.axes.c2p(0,0,3.2*self.dt) + 0.5*RIGHT)
        self.add(t)

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

        self.add(*self.grids.values())

        # # self.wait(2)
        # return

        # create processes as circles
        self.processes = []
        for i in range(self.num_space * self.num_timesteps):
            c = self.create_process_circle(i)
            self.processes.append(c)

        procs = VGroup(*self.processes).arrange_in_grid(cols=int(self.num_space * self.num_timesteps / 3))
        # put box around them
        world_box = SurroundingRectangle(procs)
        procs.add(world_box)
        procs.to_edge(LEFT+DOWN)

        self.add_fixed_in_frame_mobjects(procs)
        self.remove(procs)


        ## STEP: show mpi://WORLD
        self.update_text(f"mpi://WORLD with\n{self.num_space * self.num_timesteps} processes",
                         t2c={'mpi://WORLD': YELLOW, f"{self.num_space * self.num_timesteps}": GREEN})
        self.play(FadeIn(self.cur_text), FadeIn(procs))

        procs.remove(world_box)
        self.play(FadeOut(self.cur_text), FadeOut(world_box))


        ### STEP: split up mpi://WORLD
        self.update_text(f"Create pset grid")

        self.play(FadeIn(self.cur_text),
                  procs.animate.arrange_in_grid(cols=self.num_space, rows=self.num_timesteps, flow_order='ul').to_edge(LEFT+DOWN))
        self.play(FadeOut(self.cur_text))

        # ungroup processes and box
        self.remove(procs)
        procs.remove(world_box)
        procs.remove(*self.processes)

        # keep processes in the view
        self.add(*self.processes)

        ## STEP: show space psets
        # create boxes around each set
        space_sets = [VGroup(*self.processes[i::self.num_timesteps]) for i in range(self.num_timesteps)]
        space_boxes = []
        for i,s in enumerate(space_sets):
            box = SurroundingRectangle(s, color=self.space_pset_color, fill_color=self.space_pset_color, fill_opacity=0.2)
            self.add_fixed_in_frame_mobjects(box)
            self.remove(box)
            self.bring_to_back(box)
            space_boxes.append(box)

        self.update_text(f"Space psets")
        self.play(*[FadeIn(box) for box in space_boxes], FadeIn(self.cur_text))
        # self.play(*[FadeOut(box) for box in space_boxes], FadeOut(self.cur_text))
        self.play(FadeOut(self.cur_text))



        ## STEP: show time psets
        # create boxes around each set
        split_sets = [VGroup(*self.processes[i*self.num_timesteps:(i+1)*self.num_timesteps][::-1]) for i in range(self.num_space)]
        split_boxes = []
        for i,s in enumerate(split_sets):
            box = SurroundingRectangle(s, color=self.split_colors[i], fill_color=self.split_colors[i], fill_opacity=0.7)
            self.add_fixed_in_frame_mobjects(box)
            self.remove(box)
            self.bring_to_back(box)
            split_boxes.append(box)

        self.update_text(f"Time psets\n → LibPFASST instances")
        self.play(*[FadeIn(box) for box in split_boxes], FadeIn(self.cur_text))
        self.play(*[FadeOut(box) for box in split_boxes], *[FadeOut(box) for box in space_boxes], FadeOut(self.cur_text))


        ## Spread them a bit
        animations = []
        for i,s in enumerate(split_sets[::-1]):
            # # move sets side to side
            animations.append(s.animate.shift(RIGHT*0.5*i))


        self.play(*animations)


        #  ungroup processes
        for s in space_sets:
            s.remove(*s)

        # remove processes
        self.play(*[FadeOut(p) for p in self.processes])

        # put into 3d space
        for i,p in enumerate(self.processes):
            p.unfix_from_frame()

        # move processes to their position on the axis
        animations = []
        for i,proc in enumerate(self.processes):
            pos = np.array(self.get_proc_pos_c(i, self.t0, self.num_timesteps))
            proc.move_to(self.axes.c2p(*pos))
            proc.rotate(180*DEGREES, axis=OUT)

        self.update_text("Turn each time pset\ninto a PFASST instance")
        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in self.processes])

        ## STEP: highlight pfasst psets
        # show pset boxes shortly
        psets = self.create_pfasst_pset_boxes(self.t0, self.num_timesteps)

        self.play(*[FadeIn(p) for p in psets])
        self.play(*[FadeOut(p) for p in psets], FadeOut(self.cur_text))


        ## STEP: show space communicators
        self.update_text(f"{self.num_timesteps} space psets")
        # arrows = []
        space_psets = self.create_space_pset_boxes(self.t0 + self.cur_step*self.dt, self.num_timesteps)


        # for some readon, fadein does not work for these
        self.add(*space_psets)
        self.play(FadeIn(self.cur_text))
        # self.play(FadeOut(self.cur_text), *[FadeOut(a) for a in arrows])
        self.play(FadeOut(self.cur_text), *[FadeOut(p) for p in space_psets])



        ## STEP: show initial condition
        panes = [self.get_value_pane(self.t0) for i in range(self.num_timesteps+1)]

        # fix position for some of the things
        for i,pane in enumerate(panes[:-1]):
            pos = self.get_valuepane_pos(self.t0+(self.cur_step + i)*self.dt)
            pane.move_to(self.axes.c2p(*pos))

        self.update_text("Setup Initial Condition")


        # don't hide grid
        for pane in panes:
            self.bring_to_back(pane)

        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in panes[:self.num_timesteps]])
        self.play(FadeOut(self.cur_text))


        ## STEP: solve pfasst block
        self.update_text("Solve PFASST block")

        for p in panes[1:]:
            self.remove(p)


        # put solution everwhere
        panes[1:] = [self.get_value_pane(self.t0+i*(self.dt)) for i in range(1,self.num_timesteps+1)]

        # don't hide grid
        for p in panes[1:]:
            self.bring_to_back(p)

        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in panes[1:]])
        self.play(FadeOut(self.cur_text))



        ## STEP: spread solution
        self.update_text("Bcast solution for next\ntime block")

        # remove panes
        self.play(*[FadeOut(p) for p in panes[:-1]], FadeIn(self.cur_text))
        # copy last pane to rest
        for i in range(len(panes)-1):
            panes[i] = panes[-1].copy()
            self.bring_to_back(panes[i])
        # move each pane to its position
        animations = []
        for i,p in enumerate(panes[:-1]):
            pos = self.get_valuepane_pos(self.t0+(i)*self.dt)
            animations.append(p.animate.move_to(self.axes.c2p(*pos)))

        # remove last pane
        self.remove(panes[-1])
        panes = panes[:-1]
        self.play(*animations, FadeOut(self.cur_text))


        # STEP: move everything down by number of timesteps
        self.update_text("Move to next block")
        self.play(FadeIn(self.cur_text))
        animations = []
        for g in self.grids.values():
            animations.append(g.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes_labels.animate.shift(IN*self.dt_p*self.num_timesteps))

        # remove panes for the zoom out
        # remove panes as they can't be scaled in opengl
        animations += [FadeOut(p) for p in panes]


        self.play(*animations, run_time=2)
        self.play(FadeOut(self.cur_text))

        self.cur_step += self.num_timesteps



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



        ## STEP: show old psets
        animations = []
        old_split_boxes = self.create_pfasst_pset_boxes(self.t0 + self.cur_step*self.dt, self.num_timesteps)
        for box in old_split_boxes:
            # make small gap with the boxes
            height = (box.get_zenith() - box.get_nadir())[2]
            box.stretch(0.90, 2)
            new_height = (box.get_zenith() - box.get_nadir())[2]
            box.shift(IN*(height-new_height)/2)
            animations.append(FadeIn(box))

        self.update_text("Case 1:\nAddition of new processes")
        self.play(*animations, FadeIn(self.cur_text))
        self.play(FadeOut(self.cur_text))


        self.update_text("Runtime:\nnew set mpi://add_0", t2c={'mpi://add_0': YELLOW})

        # # create processes as circles
        num_new_timesteps = 2
        new_processes = []
        for i in range(num_new_timesteps*self.num_space):
            c = Circle(radius=0.2, fill_color=GREEN, fill_opacity=0.7)
            t = Tex(f"{i+len(self.processes)}").scale(0.7)
            t.move_to(c)
            v = VGroup(c, t)
            v.scale(0.6)
            new_processes.append(v)

        procs = VGroup(*new_processes[::-1]).arrange_in_grid(cols=int(self.num_space * self.num_timesteps / 3))
        # put box around them
        add_box = SurroundingRectangle(procs)
        procs.add(add_box)

        procs.rotate(90*DEGREES, axis=RIGHT)
        procs.to_edge(LEFT*UP)
        procs.shift(DOWN*4)
        self.play(FadeIn(self.cur_text), FadeIn(procs))
        self.play(FadeOut(self.cur_text))


        # ungroup processes and box
        self.remove(procs)
        procs.remove(add_box)
        procs.remove(*new_processes)
        # keep processes in the view
        self.add(*new_processes)


        ## STEP: split up new procs
        animations = []
        split_sets = [VGroup(*new_processes[i*num_new_timesteps:(i+1)*num_new_timesteps][::-1]) for i in range(self.num_space)]
        for i in range(self.num_space):
            # add box around each set
            # need to flip it onto the xy pane to make box right
            split_sets[i].rotate(-90*DEGREES, axis=RIGHT)
            split_sets[i].arrange_in_grid(cols=1)
            box = SurroundingRectangle(split_sets[i], color=self.split_colors[i])
            split_sets[i].add(box)
            split_sets[i].rotate(90*DEGREES, axis=RIGHT)

            # move sets side to side
            split_sets[i].generate_target()
            split_sets[i].target.shift(LEFT*0.5*i)

            animations.append(MoveToTarget(split_sets[i]))

        self.update_text("Split up new pset", t2c={'mpi://add_0': YELLOW})
        self.play(*animations, FadeIn(self.cur_text))


        ## STEP: move procs to position
        # flip processes horizontally
        animations = [p.animate.rotate(-90 * DEGREES, axis=RIGHT) for p in new_processes]
        for i in range(self.num_space):
            # remove boxes
            animations.append(FadeOut(split_sets[i][-1]))
            split_sets[i].remove(split_sets[i][-1])

        self.play(*animations, FadeOut(self.cur_text))



        # move processes to their position on the axis
        animations = []
        for i,proc in enumerate(new_processes):
            proc.generate_target()
            pos = np.array(self.get_proc_pos_c(i, self.t0 + (self.cur_step+self.num_timesteps)*self.dt, num_new_timesteps))
            proc.target.move_to(self.axes.c2p(*pos))
            animations.append(MoveToTarget(proc))

        self.play(*animations)



        ## STEP: show union
        ##
        # self.update_text("New and old split")

        self.update_text(f"{self.num_space} union operations")

        # create boxes for new splits
        new_split_boxes = self.create_pfasst_pset_boxes(self.t0 + (self.cur_step + self.num_timesteps)*self.dt, num_new_timesteps)

        animations = []
        for box in new_split_boxes:
            # height = (box.get_zenith() - box.get_nadir())[2]
            # box.stretch(0.90, 2)
            # new_height = (box.get_zenith() - box.get_nadir())[2]
            # box.shift(OUT*(height-new_height)/2)
            animations.append(FadeIn(box))


        self.play(*animations, FadeIn(self.cur_text))
        # self.play(FadeOut(self.cur_text))



        joined_split_boxes = self.create_pfasst_pset_boxes(self.t0+1*self.num_timesteps*self.dt, self.num_timesteps+num_new_timesteps)

        animations = []
        for box1,box2,joined in zip(new_split_boxes, old_split_boxes, joined_split_boxes):
            animations += [FadeOut(box1), FadeOut(box2), FadeIn(joined)]
        self.play(*animations)

        animations = []
        for box in joined_split_boxes:
            animations.append(FadeOut(box))

        self.play(*animations)
        self.play(FadeOut(self.cur_text))



        # update some vars
        self.num_timesteps += num_new_timesteps
        self.processes += new_processes


        # ## STEP: show initial condition
        # panes = [self.get_value_pane(self.cur_step * self.dt) for i in range(self.num_timesteps+1)]

        # # fix position for some of the things
        # for i,pane in enumerate(panes[:-1]):
        #     pos = self.get_valuepane_pos(self.t0+(self.cur_step + i)*self.dt)
        #     pane.move_to(self.axes.c2p(*pos))


        # # don't hide grid
        # for pane in panes:
        #     self.bring_to_back(pane)

        # self.play(*[FadeIn(p) for p in panes[:self.num_timesteps]])
        # # self.play(FadeOut(self.cur_text))


        # ## STEP: solve pfasst block
        # for p in panes[1:]:
        #     self.remove(p)

        # put solution everwhere
        panes[0:] = [self.get_value_pane(self.t0+(self.cur_step+i)*self.dt) for i in range(0,self.num_timesteps+1)]

        # don't hide grid
        for p in panes[0:]:
            self.bring_to_back(p)

        self.play(*[FadeIn(p) for p in panes[0:]])


        ## STEP: spread solution
        # remove panes
        self.play(*[FadeOut(p) for p in panes[:-1]])
        # copy last pane to rest
        for i in range(len(panes)-1):
            panes[i] = panes[-1].copy()
            self.bring_to_back(panes[i])
        # move each pane to its position
        animations = []
        for i,p in enumerate(panes[:-1]):
            pos = self.get_valuepane_pos(self.t0+(self.cur_step + i)*self.dt)
            animations.append(p.animate.move_to(self.axes.c2p(*pos)))

        # remove last pane
        self.remove(panes[-1])
        panes = panes[:-1]
        self.play(*animations)


        # STEP: move everything down by number of timesteps
        self.update_text("Move to next block")
        self.play(FadeIn(self.cur_text))
        animations = []
        for g in self.grids.values():
            animations.append(g.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes_labels.animate.shift(IN*self.dt_p*self.num_timesteps))

        # remove panes for the zoom out
        # remove panes as they can't be scaled in opengl
        animations += [FadeOut(p) for p in panes]


        self.play(*animations, run_time=2)
        self.play(FadeOut(self.cur_text))

        self.cur_step += self.num_timesteps

        # self.interactive_embed()
        self.wait(2)


    def update_text(self, new_text, **kwargs):
        new_text = Text(new_text, **kwargs).scale(0.7)
        new_text.to_edge(UP+LEFT)
        # need to add and remove to be able to have it in a fixed in frame with effect
        self.add_fixed_in_frame_mobjects(new_text)
        self.remove(new_text)
        self.cur_text = new_text

    def get_valuepane_pos(self, timestep):
        return np.array([self.sidelen_c/2,self.sidelen_c/2,timestep])


    def create_process_circle(self, i):
        c = Circle(radius=0.25/self.cur_axes_scale, fill_color=GREEN, fill_opacity=0.7)
        t = Tex(f"{i}").scale(0.8/self.cur_axes_scale)
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
