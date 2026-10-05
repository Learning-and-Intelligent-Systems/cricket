#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr char* name = "{{lower(name)}}";
    static constexpr std::size_t dimension = {{n_q}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr char* end_effector = "{{end_effector}}";

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

    template <std::size_t rake>
    struct Spheres
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_m{
        {{join(bound_range, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_a{
        {{join(bound_lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> d_m{
        {{join(bound_descale, ", ")}}
    };

    static inline void scale_configuration(Configuration& q) noexcept
    {
        q = q * Configuration(s_m) + Configuration(s_a);
    }

    static inline void descale_configuration(Configuration& q) noexcept
    {
        q = (q - Configuration(s_a)) * Configuration(d_m);
    }

    template <std::size_t rake>
    static inline void scale_configuration_block(ConfigurationBlock<rake> &q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_lower, index) }} + (q[{{index}}] * {{ at(bound_range, index) }});
        {%- endfor %}
    }

    template <std::size_t rake>
    static inline void descale_configuration_block(ConfigurationBlock<rake> & q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_descale, index) }} * (q[{{index}}] - {{ at(bound_lower, index) }});
        {%- endfor %}
    }

    inline static auto space_measure() noexcept -> float
    {
        return {{measure}};
    }

    template <std::size_t rake>
    static inline void sphere_fk(const ConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{spherefk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{spherefk_code_output}}> y;

        {{spherefk_code}}

        for (auto i = 0U; i < {{n_spheres}}; ++i)
        {
            out.x[i] = y[i * 4 + 0];
            out.y[i] = y[i * 4 + 1];
            out.z[i] = y[i * 4 + 2];
            out.r[i] = y[i * 4 + 3];
        }
    }

    // Out-of-line sphere-set tests for the cluster level (ccfk_template_clustered.hh):
    // `a`/`b` hold sphere indices times 4 into the FK output array y.
    template <typename DataT, typename Y>
    [[gnu::noinline]] static auto sphere_set_environment(
        const vamp::collision::Environment<DataT> &environment,
        const Y &y,
        const std::size_t *a,
        std::size_t n) noexcept -> bool
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto s = a[i];
            if (sphere_environment_in_collision(environment, y[s + 0], y[s + 1], y[s + 2], y[s + 3]))
            {
                return true;
            }
        }
        return false;
    }

    template <typename DataT, typename Y>
    [[gnu::noinline]] static auto sphere_set_attachment(
        const vamp::collision::Environment<DataT> &environment,
        const Y &y,
        const std::size_t *a,
        std::size_t n) noexcept -> bool
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto s = a[i];
            if (attachment_sphere_collision<typename Y::value_type>(environment, y[s + 0], y[s + 1], y[s + 2], y[s + 3]))
            {
                return true;
            }
        }
        return false;
    }

    template <typename Y>
    [[gnu::noinline]] static auto sphere_sets_collide(
        const Y &y,
        const std::size_t *a,
        std::size_t na,
        const std::size_t *b,
        std::size_t nb) noexcept -> bool
    {
        for (std::size_t i = 0; i < na; ++i)
        {
            const auto p = a[i];
            for (std::size_t j = 0; j < nb; ++j)
            {
                const auto q = b[j];
                if (sphere_sphere_self_collision<typename Y::value_type>(
                        y[p + 0], y[p + 1], y[p + 2], y[p + 3], y[q + 0], y[q + 1], y[q + 2], y[q + 3]))
                {
                    return true;
                }
            }
        }
        return false;
    }

    template <typename Y>
    [[gnu::noinline]] static void sphere_sets_debug(
        const Y &y,
        const std::size_t *a,
        std::size_t na,
        const std::size_t *b,
        std::size_t nb,
        std::vector<std::pair<std::size_t, std::size_t>> &out)
    {
        for (std::size_t i = 0; i < na; ++i)
        {
            for (std::size_t j = 0; j < nb; ++j)
            {
                const auto p = a[i] * 4, q = b[j] * 4;
                if (sphere_sphere_self_collision<typename Y::value_type>(
                        y[p + 0], y[p + 1], y[p + 2], y[p + 3], y[q + 0], y[q + 1], y[q + 2], y[q + 3]))
                {
                    out.emplace_back(a[i], b[j]);
                }
            }
        }
    }

    using Debug = std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<std::size_t, std::size_t>>>;

    template <std::size_t rake>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept -> Debug
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}

        Debug output;

        for (auto i = 0U; i < {{ n_spheres }}; ++i)
        {
            output.first.emplace_back(
                sphere_environment_get_collisions<decltype(x[0])>(
                    environment, y[i * 4 + 0], y[i * 4 + 1], y[i * 4 + 2], y[i * 4 + 3]));
        }

        // robot self-collisions: one loop per allowed link pair (the sphere
        // index tables are compile-time constants; code size O(pairs), not
        // O(sphere pairs))
        {% for i in range(length(allowed_link_pairs)) %}
        {% set pair = at(allowed_link_pairs, i) %}
        {% set link_1_spheres = at(per_link_spheres, at(pair, 0)) %}
        {% set link_2_spheres = at(per_link_spheres, at(pair, 1)) %}
        {
            static constexpr std::array<std::size_t, {{ length(link_1_spheres) }}> s1{ {% for j in link_1_spheres %}{{ j }}{% if not loop.is_last %}, {% endif %}{% endfor %} };
            static constexpr std::array<std::size_t, {{ length(link_2_spheres) }}> s2{ {% for k in link_2_spheres %}{{ k }}{% if not loop.is_last %}, {% endif %}{% endfor %} };
            sphere_sets_debug(y, s1.data(), s1.size(), s2.data(), s2.size(), output.second);
        }
        {% endfor %}

        return output;
    }

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}
        {% include "ccfk" %}

        return true;
    }

    template <std::size_t rake>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfkee_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfkee_code_output}}> y;

        {{ccfkee_code}}
        {% include "ccfk" %}

        // attaching at {{ end_effector }}
        set_attachment_pose(environment, to_isometry(&y[{{ccfkee_code_output - 12}}]));

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment))
        {
            return false;
        }

        //
        // attachment vs. robot collisions
        //

        {% for i in range(length(end_effector_collisions)) %}
        {% set link_index = at(end_effector_collisions, i) %}
        {% set link_bs = at(bounding_sphere_index, link_index) %}
        {% set link_clusters = at(clusters, link_index) %}
        // Attachment vs. {{ at(link_names, link_index )}}
        if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                        y[{{(n_spheres + link_bs) * 4 + 0}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 1}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 2}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 3}}]))
        {
        {% for c in link_clusters %}
        {% if length(link_clusters) > 1 %}
            if (attachment_sphere_collision<decltype(x[0])>(environment, y[{{ c.anchor * 4 + 0 }}], y[{{ c.anchor * 4 + 1 }}], y[{{ c.anchor * 4 + 2 }}], {{ c.radius }}F))
        {% endif %}
            {
                static constexpr std::array<std::size_t, {{ length(c.spheres) }}> sa{ {% for s in c.spheres %}{{ s * 4 }}{% if not loop.is_last %}, {% endif %}{% endfor %} };
                if (sphere_set_attachment(environment, y, sa.data(), sa.size())) { return false; }
            }
        {% endfor %}
        }
        {% endfor %}
        return true;
    }

    static inline auto eefk(const std::array<float, {{n_q}}> &x) noexcept -> Eigen::Isometry3f
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometry(y.data());
    }
};
}

// NOLINTEND(*-magic-numbers)
