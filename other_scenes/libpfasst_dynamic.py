from manim import *
from manim.opengl import *

import matplotlib.pyplot as plt
import numpy as np




class HeatEqu(ThreeDScene):
    def construct(self):
        phi_deg = 70
        theta_deg = 30
        self.set_camera_orientation(phi=phi_deg * DEGREES, theta=theta_deg * DEGREES)

        self.t0 = 0
        self.num_timesteps = 3
        self.num_space = 4 # must be square number
        self.dt = 0.005

        self.sidelen_c = np.pi

        self.num_space_side = int(np.sqrt(self.num_space))

        self.plot_resolution = 0.2

        self.z_min = -10*self.dt
        self.z_max = 10*self.dt

        self.split_colors = [RED, BLUE, GREEN, PURPLE, ORANGE, PINK, YELLOW]
        if self.num_space > 1:
            # create cycle
            self.split_colors *= int(self.num_space / len(self.split_colors)) + 1
        self.split_colors = self.split_colors[:self.num_space]

        self.axes = ThreeDAxes(
                x_range=[-1.5*self.sidelen_c,1.5*self.sidelen_c,1],
                x_length=11,
                y_range=[-1.5*self.sidelen_c,1.5*self.sidelen_c,1],
                y_length=11,
                z_range=[self.z_min,self.z_max,self.dt/5],
                z_length=3*(self.z_max-self.z_min)/self.dt,
                )

        self.axes.scale(0.7)

        self.sidelen_p = (self.axes.c2p(self.sidelen_c,0,0)-self.axes.c2p(0,0,0))[0]
        self.dt_p = (self.axes.c2p(0,0,self.dt) - self.axes.c2p(0,0,0))[2]
        self.axes.shift(3*IN)

        self.axes_labels = self.axes.get_axis_labels(
                "x", "y", "t"
                )

        # self.add_fixed_orientation_mobjects(axes)

        # self.add_fixed_orientation_mobjects(labels)
        self.add(self.axes, self.axes_labels)


        t = Text("Time →")
        t.rotate(90*DEGREES, axis=RIGHT)
        t.rotate(270*DEGREES, axis=UP)
        t.scale(0.4)
        self.add(t)
        t.move_to(self.axes.c2p(0,0,3*self.dt) + 0.5*LEFT)

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
            t_label.next_to(grid, UP)
            vg = VGroup(grid, t_label)
            self.grids[t] = vg

        self.add(*self.grids.values())


        # text2d = Text("Test").scale(0.7)
        # text2d.to_edge(UP + RIGHT)
        # self.add_fixed_orientation_mobjects(text2d)


        # create processes as circles
        self.processes = []
        for i in range(self.num_space * self.num_timesteps):
            c = Circle(radius=0.2, fill_color=GREEN, fill_opacity=0.7)
            t = Tex(f"{i}").scale(0.7)
            t.move_to(c)
            v = VGroup(c, t)
            self.processes.append(v)

        procs = VGroup(*self.processes[::-1]).arrange_in_grid(cols=int(self.num_space * self.num_timesteps / 6))
        # put box around them
        world_box = SurroundingRectangle(procs)
        procs.add(world_box)

        procs.rotate(90*DEGREES, axis=RIGHT)
        procs.to_edge(LEFT*UP)
        procs.shift(DOWN*4)


        ## STEP: show mpi://WORLD
        self.cur_text = None
        self.update_text(f"mpi://WORLD with\n{self.num_space * self.num_timesteps} processes",
                         t2c={'mpi://WORLD': YELLOW, f"{self.num_space * self.num_timesteps}": GREEN})
        self.play(FadeIn(self.cur_text), FadeIn(procs))
        self.play(FadeOut(self.cur_text), FadeOut(world_box))


        # ungroup processes and box
        self.remove(procs)
        procs.remove(world_box)
        procs.remove(*self.processes)
        # keep processes in the view
        self.add(*self.processes)


        ## STEP: split up mpi://WORLD
        animations = []
        split_sets = [VGroup(*self.processes[i*self.num_timesteps:(i+1)*self.num_timesteps]) for i in range(self.num_space)]
        for i in range(self.num_space):
            # add box around each set
            # need to flip it onto the xy plane to make box right
            split_sets[i].rotate(-90*DEGREES, axis=RIGHT)
            box = SurroundingRectangle(split_sets[i], color=self.split_colors[i])
            split_sets[i].add(box)
            split_sets[i].rotate(90*DEGREES, axis=RIGHT)

            # move sets side to side
            split_sets[i].generate_target()
            split_sets[i].target.shift(LEFT*1.5*i)

            animations.append(MoveToTarget(split_sets[i]))

        self.update_text("Split up mpi://WORLD", t2c={'mpi://WORLD': YELLOW})
        self.play(*animations, FadeIn(self.cur_text))


        ## STEP: move procs to position
        # flip processes horizontally
        animations = [p.animate.rotate(-90 * DEGREES, axis=RIGHT) for p in self.processes]
        for i in range(self.num_space):
            # remove boxes
            animations.append(FadeOut(split_sets[i][-1]))
            split_sets[i].remove(split_sets[i][-1])

        self.play(*animations, FadeOut(self.cur_text))

        # move processes to their position on the axis
        animations = []
        for i,proc in enumerate(self.processes):
            proc.generate_target()
            pos = np.array(self.get_proc_pos_c(i, self.t0))
            proc.target.move_to(self.axes.c2p(*pos))
            animations.append(MoveToTarget(proc))


        ## STEP: highlight pfasst psets
        self.update_text("Turn each split into\na PFASST instance")
        self.play(FadeIn(self.cur_text), *animations)

        # show pset boxes shortly
        psets = self.create_pfasst_pset_boxes(self.t0, self.num_timesteps)

        self.play(*[FadeIn(p) for p in psets])
        self.play(*[FadeOut(p) for p in psets], FadeOut(self.cur_text))


        ## STEP: show space communicators
        self.update_text(f"{self.num_timesteps} space communicators")
        arrows = []
        for i in range(self.num_timesteps):
            # create arrow that points to that communicator
            arrow = Arrow(start=self.axes.c2p(self.sidelen_c+2, self.sidelen_c/2, self.t0+i*self.dt),
                          end=self.axes.c2p(self.sidelen_c, self.sidelen_c/2, self.t0+i*self.dt),
                          stroke_width=10)
            arrows.append(arrow)
        self.play(FadeIn(self.cur_text), *[FadeIn(a) for a in arrows])
        self.play(FadeOut(self.cur_text), *[FadeOut(a) for a in arrows])



        ## STEP: show initial condition
        planes = [self.get_value_plane(self.t0+i*(self.dt)) for i in range(self.num_timesteps+1)]

        self.update_text("Setup Initial Condition")


        # don't hide grid
        self.bring_to_back(planes[0])

        self.play(FadeIn(self.cur_text), FadeIn(planes[0]))
        self.play(FadeOut(self.cur_text))


        ## STEP: solve pfasst block
        self.update_text("Solve PFASST block")

        # don't hide grid
        for p in planes[1:]:
            self.bring_to_back(p)

        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in planes[1:]])
        self.play(FadeOut(self.cur_text))



        ## STEP: spread solution
        self.update_text("Spread solution")

        # remove planes
        self.play(*[FadeOut(p) for p in planes[:-1]], FadeIn(self.cur_text))
        # copy last plane to rest
        for i in range(len(planes)-1):
            planes[i] = planes[-1].copy()
            self.bring_to_back(planes[i])
        # move each plane to its position
        animations = []
        for i,p in enumerate(planes[:-1]):
            pos = self.get_valueplane_pos(self.t0+(i)*self.dt)
            animations.append(p.animate.move_to(self.axes.c2p(*pos)))
        self.play(*animations, FadeOut(self.cur_text))


        # STEP: move everything down by number of timesteps
        self.update_text("Move to next block")
        self.play(FadeIn(self.cur_text))
        animations = []
        for g in self.grids.values():
            animations.append(g.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes.animate.shift(IN*self.dt_p*self.num_timesteps))
        animations.append(self.axes_labels.animate.shift(IN*self.dt_p*self.num_timesteps))
        self.play(*animations, run_time=2)
        self.play(FadeOut(self.cur_text))



        ## STEP: solve next block
        self.remove(*planes[1:])
        planes = [self.get_value_plane(self.t0+(i+self.num_timesteps)*(self.dt)) for i in range(self.num_timesteps+1)]

        self.update_text("Solve next block")

        # don't hide grid
        for p in planes[1:]:
            self.bring_to_back(p)

        self.play(FadeIn(self.cur_text), *[FadeIn(p) for p in planes[1:]])
        self.play(FadeOut(self.cur_text))


        # self.embed()
        self.wait(2)


    def update_text(self, new_text, **kwargs):
        new_text = Text(new_text, **kwargs).scale(0.7).shift(LEFT*4 + UP*3)
        # need to add and remove to be able to have it in a fixed in frame with effect
        self.add_fixed_in_frame_mobjects(new_text)
        self.remove(new_text)
        self.cur_text = new_text

    def get_valueplane_pos(self, timestep):
        return np.array([self.sidelen_c/2,self.sidelen_c/2,timestep])

    def get_value_plane(self, timestep):
        im = OpenGLImageMobject(get_heat_image(0, self.sidelen_c, 0, self.sidelen_c, self.plot_resolution, timestep))
        im.scale(self.sidelen_p/im.width)
        im.move_to(self.axes.c2p(*self.get_valueplane_pos(timestep)))

        return im

    def get_proc_pos_c(self, p_num, base_t):
        """
        get proc position in axis coordinates
        """
        # get time step
        t = base_t + self.dt*(p_num%self.num_timesteps)
        i = p_num//self.num_timesteps

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

            # # cylinder = Cylinder(height=dt_p*(num_timesteps-1), radius=0.2, fill_color=split_colors[i], fill_opacity=0.2, stroke_width=0)
            # cylinder = Cylinder(radius=2, height=1)
            # psets.append(cylinder)

            pos = self.get_proc_pos_c(i*self.num_timesteps, base_t)
            # fix z position
            pos[2] = self.dt*(base_t + num_timesteps_height/2)
            psets[-1].move_to(self.axes.c2p(*pos))
        return psets


def get_heat_image(x_min, x_max, y_min, y_max, resolution, t):
    num_x = int((x_max-x_min)/resolution) - 1
    num_y = int((y_max-y_min)/resolution) - 1
    im = np.zeros((num_y,num_x))
    for j in range(num_y):
        for i in range(num_x):
            x = x_min + (i+0.5)*resolution
            y = y_min + (j+0.5)*resolution
            im[j,i]  =   (np.sin(x) * np.exp(-1.0 * np.pi**2.0 * t)) \
                       * (np.sin(y) * np.exp(-1.0 * np.pi**2.0 * t))

    # no need to normalize as max value is exactly 1
    cmap = plt.get_cmap('viridis')
    rgba_im = cmap(im)
    rgb_im = np.delete(rgba_im, 3, 2)
    
    # convert to uint8
    rgb_im = np.uint8(rgb_im * 256)
    return rgb_im
